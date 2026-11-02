# CLEAN_MF_N32_3d.py -- same code as CLEAN_MF_N32_2d.py; makes only the 3D plots (reuses the 2D run's checkpoint).
import sys
import os
import time

# One BLAS thread per process (rows are parallelized over processes); must be set before numpy.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy import optimize
import matplotlib
import importlib as _importlib


def _select_interactive_backend():
    """Use the first available GUI backend (Qt, then Tk) so 3D plots can be rotated; None if none."""
    candidates = [
        ('QtAgg', ('PyQt6', 'PySide6', 'PyQt5', 'PySide2')),
        ('TkAgg', ('tkinter',)),
    ]
    for backend_name, modules in candidates:
        for mod_name in modules:
            try:
                _importlib.import_module(mod_name)
            except ImportError:
                continue
            try:
                matplotlib.use(backend_name)
            except Exception:
                continue
            return backend_name
    return None


# An explicitly requested backend (e.g. `MPLBACKEND=Agg python3 CLEAN_MF_N32_3d.py` for a
# background run without plot windows) is respected; otherwise pick an interactive one.
if os.environ.get('MPLBACKEND'):
    _chosen_backend = os.environ['MPLBACKEND']
else:
    _chosen_backend = _select_interactive_backend()
if _chosen_backend is None:
    print('No interactive GUI toolkit (Qt or Tk) found in this Python environment -- using '
          'matplotlib\'s own default backend (%r) instead. All plots will still save to PNG '
          'normally; plt.show() just will not open a rotatable window. To get interactive 3D '
          'rotation, install ONE GUI toolkit into THIS SAME Python environment PyCharm is '
          'using (the one shown in this traceback\'s file paths) -- easiest via PyCharm\'s '
          'Settings -> Project -> Python Interpreter -> "+" -> search "PyQt5" -> Install, '
          'rather than a system-wide pip install, which would land in a different '
          'environment and not be seen here.' % matplotlib.get_backend())
else:
    print('matplotlib backend in use:', _chosen_backend)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection,
                                          # needed for fig.add_subplot(111, projection='3d')
import pickle
from dataclasses import dataclass


@dataclass
class Parameters:
    """Model parameters of the zero-field Kitaev-Heisenberg-Gamma model on an Nx x Ny torus."""
    Nx: int
    Ny: int
    Kxyz: np.ndarray    # (Kx, Ky, Kz) Kitaev couplings on the x-, y-, z-bonds
    Jxyz: np.ndarray    # (Jx, Jy, Jz) Heisenberg couplings of the S^x S^x, S^y S^y, S^z S^z terms
    Gamma: float        # symmetric off-diagonal coupling


# Number of independent mean-field unknowns (24 mean-field parameters + 3 Lagrange multipliers);
# see MF_PARAMETER_NAMES further below for their order.
N_MF_PARAMETERS = 27

# Appended to the name of every saved plot (just before '.png'), so figures made with this
# updated zero-field version never overwrite the ones from the earlier field-dependent script.
PLOT_NAME_TAG = '_up'
# Prepended to the FILE name (not the folder) of every saved plot, marking figures made with this
# clean zero-field mean-field code: e.g. CLEAN_MF_phase_diagram_JperpJz_Gamma0_phase_up.png.
PLOT_NAME_PREFIX = 'CLEAN_MF_'


def _clean_mf_name(path):
    """Return `path` with PLOT_NAME_PREFIX put in front of its file name (once)."""
    folder, name = os.path.split(path)
    if not name.startswith(PLOT_NAME_PREFIX):
        name = PLOT_NAME_PREFIX + name
    return os.path.join(folder, name)


class system_parameters(object):
    # Majorana boundary conditions: k = 2*pi*(n + shift)/N; shift 0 = periodic, 0.5 = antiperiodic.
    kx_shift = 0.0
    ky_shift = 0.0

    def __init__(self, parameters, dtype=np.complex128):
        self.Nx = parameters.Nx
        self.Ny = parameters.Ny

        self.kx_basis = 2 * np.pi / self.Nx
        self.ky_basis = 2 * np.pi / self.Ny

        self.MF_parameters = None

        self.dtype = dtype

    def save_type_name(self):

        save_directory = (f'{os.getcwd()}/paper_plots/'
                          f'K{[round(float(k), 6) for k in np.atleast_1d(self.Kxyz)]}'
                          f'J{[round(float(j), 6) for j in np.atleast_1d(self.Jxyz)]}'
                          f'Gamma{round(float(self.Gamma), 6)}/')

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        self.save_path = f'{save_directory}Nx{self.Nx}'


    def get_eiegensystem_Majorana(self):

        eigs = np.zeros(shape=(self.Nx, self.Ny, 8), dtype=self.dtype)
        vecs = np.zeros(shape=(self.Nx, self.Ny, 8, 8), dtype=self.dtype)

        for kx in np.arange(self.Nx):
            for ky in np.arange(self.Ny):
                expkx, expky, expkz = self.momentum_function(kx, ky)
                U = np.diag(np.array([1, 1, 1, 1, 1, expkx, expky, expkz]))
                eigs[kx, ky], vecs[kx, ky] = LA.eigh(
                    np.einsum('ij,jk,kl->il', U.conj().T, self.get_Hamiltonian(kx, ky, self.MF_parameters), U))

        Ek = eigs[:, :, 4:8]
        uk = vecs[:, :, :, 4:8]

        return Ek, uk

    def GS_Energy(self):
        """Energy of the quadratic part per unit cell, -1/2 sum of positive eigenvalues / N (without C)."""
        return self._GS_Energy_general_vectorized(self.MF_parameters)

    def _GS_Energy_general_vectorized(self, MF_parameters):
        """GS_Energy for all k-points at once (one batched eigvalsh call)."""
        H = self._batched_Hamiltonian_general(MF_parameters)
        evals = np.linalg.eigvalsh(H)  # (Nx, Ny, 8), ascending -- eigenvalues only, no eigenvectors
        GSEnergy = np.sum(evals[:, :, 4:8])
        return -0.5 * float(np.real(GSEnergy)) / (self.Nx * self.Ny)

    def _batched_Hamiltonian_general(self, MF_parameters):
        """8x8 Bloch Hamiltonian at all k-points at once, shape (Nx, Ny, 8, 8)."""
        (chi_00_x, chi_00_y, chi_00_z,
         chi_xx_x, chi_yy_x, chi_zz_x,
         chi_xx_y, chi_yy_y, chi_zz_y,
         chi_xx_z, chi_yy_z, chi_zz_z,
         chi_yz_x, chi_zy_x,
         chi_xz_y, chi_zx_y,
         chi_xy_z, chi_yx_z,
         mx_A, my_A, mz_A, mx_B, my_B, mz_B,
         lagrange_x, lagrange_y, lagrange_z) = MF_parameters

        Kx, Ky, Kz = self.Kxyz
        Jx, Jy, Jz = self.Jxyz
        Gamma = self.Gamma

        kx_idx = np.arange(self.Nx).reshape(-1, 1)
        ky_idx = np.arange(self.Ny).reshape(1, -1)
        expkx = np.broadcast_to(np.exp(1j * self.kx_basis * (kx_idx + self.kx_shift)), (self.Nx, self.Ny)).astype(self.dtype)
        expky = np.broadcast_to(np.exp(1j * self.ky_basis * (ky_idx + self.ky_shift)), (self.Nx, self.Ny)).astype(self.dtype)

        H = np.zeros((self.Nx, self.Ny, 8, 8), dtype=self.dtype)

        term_x = (Kx + Jx) * chi_xx_x + Jy * chi_yy_x + Jz * chi_zz_x \
            + Gamma * chi_yz_x + Gamma * chi_zy_x
        term_y = (Ky + Jy) * chi_yy_y + Jx * chi_xx_y + Jz * chi_zz_y \
            + Gamma * chi_xz_y + Gamma * chi_zx_y
        term_z = (Kz + Jz) * chi_zz_z + Jx * chi_xx_z + Jy * chi_yy_z \
            + Gamma * chi_xy_z + Gamma * chi_yx_z

        H[:, :, 0, 4] += -0.25j * (term_x * expkx + term_y * expky + term_z)

        xx = (Kx + Jx) * chi_00_x + Jx * chi_00_y * expky * np.conjugate(expkx) + Jx * chi_00_z * np.conjugate(expkx)
        yy = (Ky + Jy) * chi_00_y + Jy * chi_00_x * expkx * np.conjugate(expky) + Jy * chi_00_z * np.conjugate(expky)
        zz = (Kz + Jz) * chi_00_z + Jz * chi_00_x * expkx + Jz * chi_00_y * expky

        H[:, :, 1, 5] += -0.25j * xx
        H[:, :, 2, 6] += -0.25j * yy
        H[:, :, 3, 7] += -0.25j * zz

        H[:, :, 2, 7] += -0.25j * chi_00_x * Gamma * expkx
        H[:, :, 3, 6] += -0.25j * chi_00_x * Gamma * expkx * np.conjugate(expky)
        H[:, :, 1, 7] += -0.25j * chi_00_y * Gamma * expky
        H[:, :, 3, 5] += -0.25j * chi_00_y * Gamma * expky * np.conjugate(expkx)
        H[:, :, 1, 6] += -0.25j * chi_00_z * Gamma * np.conjugate(expky)
        H[:, :, 2, 5] += -0.25j * chi_00_z * Gamma * np.conjugate(expkx)

        # x-bond (gamma = x)
        H[:, :, 0, 1] += 0.25j * (Jx + Kx) * mx_B
        H[:, :, 4, 5] += 0.25j * (Jx + Kx) * mx_A * np.conjugate(expkx)
        H[:, :, 0, 2] += 0.25j * (Jy * my_B + Gamma * mz_B)
        H[:, :, 4, 6] += 0.25j * (Jy * my_A + Gamma * mz_A) * np.conjugate(expky)
        H[:, :, 0, 3] += 0.25j * (Jz * mz_B + Gamma * my_B)
        H[:, :, 4, 7] += 0.25j * (Jz * mz_A + Gamma * my_A)

        # y-bond (gamma = y)
        H[:, :, 0, 1] += 0.25j * (Jx * mx_B + Gamma * mz_B)
        H[:, :, 4, 5] += 0.25j * (Jx * mx_A + Gamma * mz_A) * np.conjugate(expkx)
        H[:, :, 0, 2] += 0.25j * (Jy + Ky) * my_B
        H[:, :, 4, 6] += 0.25j * (Jy + Ky) * my_A * np.conjugate(expky)
        H[:, :, 0, 3] += 0.25j * (Jz * mz_B + Gamma * mx_B)
        H[:, :, 4, 7] += 0.25j * (Jz * mz_A + Gamma * mx_A)

        # z-bond (gamma = z)
        H[:, :, 0, 1] += 0.25j * (Jx * mx_B + Gamma * my_B)
        H[:, :, 4, 5] += 0.25j * (Jx * mx_A + Gamma * my_A) * np.conjugate(expkx)
        H[:, :, 0, 2] += 0.25j * (Jy * my_B + Gamma * mx_B)
        H[:, :, 4, 6] += 0.25j * (Jy * my_A + Gamma * mx_A) * np.conjugate(expky)
        H[:, :, 0, 3] += 0.25j * (Jz + Kz) * mz_B
        H[:, :, 4, 7] += 0.25j * (Jz + Kz) * mz_A


        # constraint (Lagrange multipliers)
        H[:, :, 0, 1] += 0.5j * lagrange_x
        H[:, :, 2, 3] += 0.5j * lagrange_x
        H[:, :, 4, 5] += 0.5j * lagrange_x * np.conjugate(expkx)
        H[:, :, 6, 7] += 0.5j * lagrange_x * expky

        H[:, :, 0, 2] += 0.5j * lagrange_y
        H[:, :, 3, 1] += 0.5j * lagrange_y
        H[:, :, 4, 6] += 0.5j * lagrange_y * np.conjugate(expky)
        H[:, :, 7, 5] += 0.5j * lagrange_y * np.conjugate(expkx)

        H[:, :, 0, 3] += 0.5j * lagrange_z
        H[:, :, 1, 2] += 0.5j * lagrange_z
        H[:, :, 4, 7] += 0.5j * lagrange_z
        H[:, :, 5, 6] += 0.5j * lagrange_z * expkx * np.conjugate(expky)

        H = H + np.conj(np.swapaxes(H, -1, -2))
        H = 2 * H
        return H

    def MF_free_energy_constant(self):
        """Constant C of the decoupling per unit cell: -1/4 sum W (m_A m_B - chi chi00)."""
        Kx, Ky, Kz = self.Kxyz
        Jx, Jy, Jz = self.Jxyz
        Gamma = self.Gamma

        (chi_00_x, chi_00_y, chi_00_z,
         chi_xx_x, chi_yy_x, chi_zz_x,
         chi_xx_y, chi_yy_y, chi_zz_y,
         chi_xx_z, chi_yy_z, chi_zz_z,
         chi_yz_x, chi_zy_x,
         chi_xz_y, chi_zx_y,
         chi_xy_z, chi_yx_z,
         mx_A, my_A, mz_A, mx_B, my_B, mz_B,
         lagrange_x, lagrange_y, lagrange_z) = self.MF_symmetry_reduction(self.MF_parameters)

        term_x = (Kx + Jx) * (mx_A * mx_B - chi_xx_x * chi_00_x) \
            + Jy * (my_A * my_B - chi_yy_x * chi_00_x) \
            + Jz * (mz_A * mz_B - chi_zz_x * chi_00_x) \
            + Gamma * (my_A * mz_B - chi_yz_x * chi_00_x) \
            + Gamma * (mz_A * my_B - chi_zy_x * chi_00_x)

        term_y = (Ky + Jy) * (my_A * my_B - chi_yy_y * chi_00_y) \
            + Jx * (mx_A * mx_B - chi_xx_y * chi_00_y) \
            + Jz * (mz_A * mz_B - chi_zz_y * chi_00_y) \
            + Gamma * (mx_A * mz_B - chi_xz_y * chi_00_y) \
            + Gamma * (mz_A * mx_B - chi_zx_y * chi_00_y)

        term_z = (Kz + Jz) * (mz_A * mz_B - chi_zz_z * chi_00_z) \
            + Jx * (mx_A * mx_B - chi_xx_z * chi_00_z) \
            + Jy * (my_A * my_B - chi_yy_z * chi_00_z) \
            + Gamma * (mx_A * my_B - chi_xy_z * chi_00_z) \
            + Gamma * (my_A * mx_B - chi_yx_z * chi_00_z)

        C = -0.25 * (term_x + term_y + term_z)
        return float(np.real(C))

    def MF_variational_energy(self):
        """Mean-field variational energy per unit cell, GS_Energy() + C (use this to compare branches)."""
        return self.GS_Energy() + self.MF_free_energy_constant()

    def plot_bandstructure(self, save='n'):

        plt.figure()  # own figure: never draw onto a still-open (e.g. 3D) figure
        energy = np.array([LA.eigh(self.get_Hamiltonian((i) % self.Nx, (-i) % self.Ny, self.MF_parameters))[0] for i in
                           np.arange(self.Nx)]).reshape(self.Nx, 8)

        for i in np.arange(4, 8):
            plt.plot(np.arange(self.Nx), np.transpose(energy[:, i]), linestyle='solid', linewidth=1,
                     color='black')  # , rcount=Nx, ccount=Ny)

        # add Kiteav limit
        energy_c = np.array([np.abs(
            0.5 * (1 + np.exp(1j * self.kx_basis * (i % self.Nx)) + np.exp(1j * self.ky_basis * ((-i) % self.Ny)))) for
                             i in np.arange(self.Nx)]).reshape(self.Nx)
        plt.plot(np.arange(self.Nx), np.transpose(energy_c), linestyle='dashed', linewidth=1,
                 color='black')  # , rcount=Nx, ccount=Ny)
        plt.hlines(y=0.262, xmin=0, xmax=self.Nx - 1, linestyle='dashed', linewidth=1, color='black')

        plt.xlabel("BZ")
        plt.ylabel('$E / |K|$')
        plt.xlim([0, self.Nx - 1])
        plt.ylim(bottom=0, top=1.8)
        plt.xticks(np.array([0, self.Nx // 3, 2 * self.Nx // 3, self.Nx - 1]),
                   [r" $\Gamma$", r" K", r" K'", r" $\Gamma$"])
        plt.yticks(np.arange(0, 2, 0.3))

        plt.gca().set_aspect(self.Nx / 3)
        if save == 'y':
            self.save_type_name()
            plt.savefig(_clean_mf_name(f'{self.save_path}_MF_spectrum{PLOT_NAME_TAG}.png'),
                        dpi=300, format='png')
        elif save == 'n':
            pass
        else:
            raise NotImplementedError('wrong input. save = y,n  ')
        plt.show()

    def _bands_along_GKKG(self, MF_parameters, n_points=None):
        """Band energies along Gamma-K-K'-Gamma for given MF parameters."""
        if n_points is None:
            n_points = self.Nx
        t = np.linspace(0, self.Nx, n_points)
        energies = np.array([LA.eigh(self.get_Hamiltonian(ti % self.Nx, (-ti) % self.Nx, MF_parameters))[0]
                              for ti in t]).reshape(n_points, 8)
        return t, energies


    def Self_Consistent_Solution(self):
        xtol = 1e-6
        maxiter = 100000

        if len(self.MF_parameters) != N_MF_PARAMETERS:
            raise ValueError(
                'self.MF_parameters has length %d; the mean-field ansatz has %d parameters. '
                'Call initialise() for a correctly sized random start, or use '
                'warm_start_from_pure_Kitaev() / build_seed().'
                % (len(self.MF_parameters), N_MF_PARAMETERS))

        x0 = self.MF_parameters

        optmz_res = sp.optimize.root(self.MF_conditions, x0, method="hybr",
                                     options={"maxfev": maxiter,
                                              "eps": 1e-12},
                                     tol=xtol)
        print(optmz_res['message'])

        self.MF_parameters = optmz_res['x']

        self.Self_Consistent_Solution_announce()

        return self.MF_parameters

    def Self_Consistent_Solution_announce(self):
        # GS_Energy() is without the constant C; MF_variational_energy() is the energy to compare.
        print("chi_00_x, chi_00_y, chi_00_z = ", self.MF_parameters[0:3])
        print("chi_xx_x, chi_yy_x, chi_zz_x (bond x) = ", self.MF_parameters[3:6])
        print("chi_xx_y, chi_yy_y, chi_zz_y (bond y) = ", self.MF_parameters[6:9])
        print("chi_xx_z, chi_yy_z, chi_zz_z (bond z) = ", self.MF_parameters[9:12])
        print("chi_yz_x, chi_zy_x (bond x offdiag) = ", self.MF_parameters[12:14])
        print("chi_xz_y, chi_zx_y (bond y offdiag) = ", self.MF_parameters[14:16])
        print("chi_xy_z, chi_yx_z (bond z offdiag) = ", self.MF_parameters[16:18])
        print("mx_A, my_A, mz_A (sublattice A) = ", self.MF_parameters[18:21])
        print("mx_B, my_B, mz_B (sublattice B) = ", self.MF_parameters[21:24])
        print("Lagrange multiplier = ", self.MF_parameters[24:27])
        GSEnergy = self.GS_Energy()
        print("GS Energy (quadratic part only, NOT the physical variational energy) = ", GSEnergy)
        print("Variational Energy (= GS Energy + constant C; USE THIS for any "
              "cross-branch/cross-solution comparison) = ", self.MF_variational_energy())

    def compute_all_MF_parameters(self, standard_normalization=False):
        """Dict of all labeled MF parameters (chi, m, lambda); standard_normalization divides chi and m by 2."""
        if self.MF_parameters is None:
            raise RuntimeError('MF_parameters is not set; run Self_Consistent_Solution() first.')

        (chi_00_x, chi_00_y, chi_00_z,
         chi_xx_x, chi_yy_x, chi_zz_x,
         chi_xx_y, chi_yy_y, chi_zz_y,
         chi_xx_z, chi_yy_z, chi_zz_z,
         chi_yz_x, chi_zy_x,
         chi_xz_y, chi_zx_y,
         chi_xy_z, chi_yx_z,
         mx_A, my_A, mz_A, mx_B, my_B, mz_B,
         lagrange_x, lagrange_y, lagrange_z) = self.MF_symmetry_reduction(self.MF_parameters)
        scale = 0.5 if standard_normalization else 1.0

        MF_values = {
            'chi^00_x': chi_00_x * scale, 'chi^00_y': chi_00_y * scale, 'chi^00_z': chi_00_z * scale,
            'chi_xx^x': chi_xx_x * scale, 'chi_yy^y': chi_yy_y * scale, 'chi_zz^z': chi_zz_z * scale,
            'chi_xx^y': chi_xx_y * scale, 'chi_yy^x': chi_yy_x * scale, 'chi_yy^z': chi_yy_z * scale,
            'chi_zz^y': chi_zz_y * scale, 'chi_xx^z': chi_xx_z * scale, 'chi_zz^x': chi_zz_x * scale,
            'chi_xy^z': chi_xy_z * scale, 'chi_xz^y': chi_xz_y * scale, 'chi_yx^z': chi_yx_z * scale, 'chi_yz^x': chi_yz_x * scale,
            'chi_zy^x': chi_zy_x * scale, 'chi_zx^y': chi_zx_y * scale,
            'lambda^x': lagrange_x, 'lambda^y': lagrange_y, 'lambda^z': lagrange_z,
            'm^x_A': mx_A * scale, 'm^y_A': my_A * scale, 'm^z_A': mz_A * scale,
            'm^x_B': mx_B * scale, 'm^y_B': my_B * scale, 'm^z_B': mz_B * scale,
            # convenience aliases: sublattice-averaged magnetization
            'm^x': 0.5 * (mx_A + mx_B) * scale, 'm^y': 0.5 * (my_A + my_B) * scale,
            'm^z': 0.5 * (mz_A + mz_B) * scale,
        }
        return MF_values

    @staticmethod
    def _is_cyclic(alpha, beta, gamma):
        """True if (alpha,beta,gamma) is a cyclic (even) permutation of (x,y,z)."""
        order = {'x': 0, 'y': 1, 'z': 2}
        idx = (order[alpha], order[beta], order[gamma])
        return idx in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    def _write_MF_block(self, f, MF_values, symbol):
        """Write one block of labeled MF parameters to an open file."""
        off_diagonal = [
            ('x', 'y', 'z', 'chi_xy^z'),
            ('x', 'z', 'y', 'chi_xz^y'),
            ('y', 'x', 'z', 'chi_yx^z'),
            ('y', 'z', 'x', 'chi_yz^x'),
            ('z', 'y', 'x', 'chi_zy^x'),
            ('z', 'x', 'y', 'chi_zx^y'),
        ]

        f.write('%s^{00}_x = %.10f\n' % (symbol, MF_values['chi^00_x']))
        f.write('%s^{00}_y = %.10f\n' % (symbol, MF_values['chi^00_y']))
        f.write('%s^{00}_z = %.10f\n' % (symbol, MF_values['chi^00_z']))
        f.write('#\n')

        for label in ['chi_xx^x', 'chi_yy^y', 'chi_zz^z',
                      'chi_xx^y', 'chi_yy^x', 'chi_yy^z',
                      'chi_zz^y', 'chi_xx^z', 'chi_zz^x']:
            alpha, gamma = label[4], label[-1]
            latex_label = '%s_{%s%s}^{%s}' % (symbol, alpha, alpha, gamma)
            f.write('%s = %.10f\n' % (latex_label, MF_values[label]))
        f.write('#\n')

        f.write('# off-diagonal (alpha != beta) b-Majorana correlators:\n')
        for alpha, beta, gamma, label in off_diagonal:
            latex_label = '%s_{%s%s}^{%s}' % (symbol, alpha, beta, gamma)
            kind = 'cyclic' if self._is_cyclic(alpha, beta, gamma) else 'anti-cyclic'
            f.write('%s = %.10f   (alpha,beta,gamma) = (%s,%s,%s) is a %s permutation of x,y,z\n'
                    % (latex_label, MF_values[label], alpha, beta, gamma, kind))
        f.write('#\n')

        f.write('# sublattice-resolved magnetization (independent A, B):\n')
        for alpha in ['x', 'y', 'z']:
            f.write('m^{%s}_A = %.10f\n' % (alpha, MF_values[f'm^{alpha}_A']))
            f.write('m^{%s}_B = %.10f\n' % (alpha, MF_values[f'm^{alpha}_B']))
        f.write('#\n')
        f.write('# sublattice-averaged magnetization (convenience alias, (m_A+m_B)/2):\n')
        for alpha in ['x', 'y', 'z']:
            f.write('m^{%s} = %.10f\n' % (alpha, MF_values[f'm^{alpha}']))
        f.write('#\n')
        for alpha in ['x', 'y', 'z']:
            f.write('\\lambda^{%s} = %.10f\n' % (alpha, MF_values[f'lambda^{alpha}']))

    def save_MF_parameters(self, filename=None, standard_normalization=False):
        """Write all MF parameters (native and standard normalization) to a text file; returns its path."""
        MF_values_native = self.compute_all_MF_parameters(standard_normalization=False)
        MF_values_std = self.compute_all_MF_parameters(standard_normalization=True)

        if filename is None:
            self.save_type_name()
            filename = f'{self.save_path}_MF_parameters.txt'

        directory = os.path.dirname(filename)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

        sections = [
            ('\\chi', MF_values_native,
             ['# Section 1: this code\'s NATIVE Majorana normalization {b,b}=2delta_{alpha beta}\n',
              '# (and likewise for c), i.e. (b^alpha)^2=1, with onsite constraint D_i = c b^x b^y b^z = 1.\n',
              '# chi^{00}_gamma = <i c_i c_j>            (c-Majorana correlator, gamma-bond)\n',
              '# chi_{alpha beta}^{gamma} = <i b^alpha_i b^beta_j>   (b-Majorana correlator, gamma-bond)\n']),
            ('u', MF_values_std,
             ['# Section 2: STANDARD Majorana normalization {gamma,gamma}=delta_{alpha beta},\n',
              '# (gamma)^2=1/2, D_i = c b^x b^y b^z = 1/4 (Kitaev 2006; You, Kimchi & Vishwanath,\n',
              '# PRB 86, 085145 (2012)), related to Section 1 by b = sqrt(2)*gamma. Since chi/u are\n',
              '# BILINEARS, Section 2 = Section 1 / 2 (NOT /sqrt(2)); lambda^alpha is identical in\n',
              '# both sections, since it is a Hamiltonian coefficient enforcing the constraint, not\n',
              '# a Majorana bilinear correlator.\n',
              '# u^{00}_gamma = <i c_i c_j> / 2                (c-Majorana correlator, gamma-bond)\n',
              '# u_{alpha beta}^{gamma} = <i b^alpha_i b^beta_j> / 2   (b-Majorana correlator, gamma-bond)\n']),
        ]
        if standard_normalization:
            sections = sections[::-1]

        with open(filename, 'w') as f:
            Kxyz_list = [float(k) for k in self.Kxyz]
            Jxyz_list = [float(j) for j in self.Jxyz]
            f.write('# Mean-field parameters for the Kitaev-Heisenberg-Gamma model (zero field)\n')
            f.write(f'# (K, J, Gamma) = ({Kxyz_list}, {Jxyz_list}, {self.Gamma})\n')
            f.write('#\n')

            for i, (symbol, MF_values, header_lines) in enumerate(sections):
                for line in header_lines:
                    f.write(line)
                f.write('#\n')
                self._write_MF_block(f, MF_values, symbol)
                if i == 0:
                    f.write('#\n#\n')

        return filename

    def Kitaev_exact_MF_parameters(self, chi_00_guess=0.5, n_iter=10, tol=1e-8, standard_normalization=False):
        """Exact MF parameters at the pure Kitaev point (chi^gg_g = 1, chi00 from the filled c band, m = lambda = 0)."""
        if not (np.allclose(self.Jxyz, 0) and self.Gamma == 0):
            raise ValueError('Kitaev_exact_MF_parameters only applies at the pure Kitaev point '
                             '(J = Gamma = 0); got Jxyz=%s, Gamma=%s' % (self.Jxyz, self.Gamma))

        # order of the 27 parameters: see MF_PARAMETER_NAMES
        chi_00_x = chi_00_y = chi_00_z = chi_00_guess
        for _ in range(n_iter):
            placeholder = np.array([
                chi_00_x, chi_00_y, chi_00_z,
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            residual = np.real(self.MF_conditions(placeholder))
            chi_00_x = residual[0] + placeholder[0]
            chi_00_y = residual[1] + placeholder[1]
            chi_00_z = residual[2] + placeholder[2]
        exact_parameters = np.array([
            chi_00_x, chi_00_y, chi_00_z,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Verify exact_parameters is genuinely a fixed point of MF_conditions (residual ~ 0
        # everywhere, not just for chi_00): confirms the topological chi/m/lambda values above.
        final_residual = np.real(self.MF_conditions(exact_parameters))
        if np.max(np.abs(final_residual)) > tol:
            raise RuntimeError('Kitaev_exact_MF_parameters: computed parameters are not a fixed '
                               'point to the requested tolerance (max residual %.3e); try more '
                               'iterations (n_iter), a different chi_00_guess, or a larger Nx, Ny.'
                               % np.max(np.abs(final_residual)))

        if standard_normalization:
            # Report-only: borrow compute_all_MF_parameters() without permanently touching
            # self.MF_parameters (restore it afterwards regardless of its prior value).
            previous_MF_parameters = self.MF_parameters
            self.MF_parameters = exact_parameters
            try:
                u_values = self.compute_all_MF_parameters(standard_normalization=True)
            finally:
                self.MF_parameters = previous_MF_parameters
            print('Kitaev_exact_MF_parameters (standard normalization {gamma,gamma}=delta, D_i=1/4):')
            print('  u^00_x, u^00_y, u^00_z = %.6f, %.6f, %.6f'
                  % (u_values['chi^00_x'], u_values['chi^00_y'], u_values['chi^00_z']))
            print('  u_a (own-bond, e.g. u_xx^x) = %.6f' % u_values['chi_xx^x'])
            print('  all cross-component / cross-bond u = 0, m^alpha = 0, lambda^alpha = 0')

        return exact_parameters

    def warm_start_from_pure_Kitaev(self, perturbation=0.01, seed=None):
        """Start vector: pure-Kitaev solution (same K, Nx, Ny) plus Gaussian noise of width `perturbation`."""
        ref_model = Majorana_rep(Parameters(self.Nx, self.Ny, self.Kxyz, np.zeros(3), 0.0))
        x0 = np.array(ref_model.Kitaev_exact_MF_parameters(), dtype=float)

        rng = np.random.default_rng(seed)
        self.MF_parameters = x0 + perturbation * rng.standard_normal(N_MF_PARAMETERS)
        return self.MF_parameters


    def momentum_function(self, kx, ky):

        expkx, expky, expkz = np.array([np.exp(1j * self.kx_basis * (kx + self.kx_shift)), np.exp(1j * self.ky_basis * (ky + self.ky_shift)), 1])

        return expkx, expky, expkz

    def MF_conditions(self, MF_parameters):
        return self.MF_conditions_general(MF_parameters)

    def get_Hamiltonian(self):

        raise NotImplementedError('Need representation & model-specific subclasses functions for Hamiltonians')


    def initialise(self):
        if self.MF_parameters is None:
            self.MF_parameters = np.random.rand(N_MF_PARAMETERS)
        elif len(self.MF_parameters) != N_MF_PARAMETERS:
            raise ValueError(
                'MF_parameters_initial has length %d but the mean-field ansatz has %d '
                'parameters. Pass MF_parameters_initial=None to get a fresh random start of '
                'the right size, or pass a %d-vector.'
                % (len(self.MF_parameters), N_MF_PARAMETERS, N_MF_PARAMETERS))

    def MF_symmetry_reduction(self, MF_parameters):
        """Return the 27 MF parameters as a tuple (order of MF_PARAMETER_NAMES)."""
        if len(MF_parameters) != N_MF_PARAMETERS:
            raise NotImplementedError('MF_parameter number incorrect (must be %d, got %d)'
                                      % (N_MF_PARAMETERS, len(MF_parameters)))
        return tuple(MF_parameters)


class Majorana_rep(system_parameters):

    def __init__(self, parameters, MF_parameters_initial=None, dtype=np.complex128):
        """Majorana mean-field model (27-parameter ansatz) for `parameters` (Nx, Ny, Kxyz, Jxyz, Gamma)."""

        super(Majorana_rep, self).__init__(parameters)

        self.Nx = parameters.Nx
        self.Ny = parameters.Ny

        self.Kxyz = parameters.Kxyz
        self.Jxyz = parameters.Jxyz
        self.Gamma = parameters.Gamma

        self.MF_parameters = MF_parameters_initial

        self.initialise()

        self.dtype = dtype

    def get_Hamiltonian(self, kx, ky, MF_parameters):
        """8x8 Majorana mean-field Bloch Hamiltonian H^c + H^b + H^bc at (kx, ky)."""
        return self._get_Hamiltonian_general(kx, ky, self.MF_symmetry_reduction(MF_parameters))


    def _get_Hamiltonian_general(self, kx, ky, MF_parameters_array):
        """8x8 Bloch Hamiltonian at one k-point from the 27 MF parameters."""
        (chi_00_x, chi_00_y, chi_00_z,
         chi_xx_x, chi_yy_x, chi_zz_x,
         chi_xx_y, chi_yy_y, chi_zz_y,
         chi_xx_z, chi_yy_z, chi_zz_z,
         chi_yz_x, chi_zy_x,
         chi_xz_y, chi_zx_y,
         chi_xy_z, chi_yx_z,
         mx_A, my_A, mz_A, mx_B, my_B, mz_B,
         lagrange_x, lagrange_y, lagrange_z) = MF_parameters_array

        Kx, Ky, Kz = self.Kxyz
        Jx, Jy, Jz = self.Jxyz
        Gamma = self.Gamma

        expkx, expky, expkz = self.momentum_function(kx, ky)

        Hamiltonian = np.zeros((8, 8), dtype=self.dtype)

        # c-Majorana hopping H^c, block [cA, cB]
        term_x = (Kx + Jx) * chi_xx_x + Jy * chi_yy_x + Jz * chi_zz_x \
            + Gamma * chi_yz_x + Gamma * chi_zy_x
        # bond y contribution (phase expky)
        term_y = (Ky + Jy) * chi_yy_y + Jx * chi_xx_y + Jz * chi_zz_y \
            + Gamma * chi_xz_y + Gamma * chi_zx_y
        # bond z contribution (phase 1, same unit cell)
        term_z = (Kz + Jz) * chi_zz_z + Jx * chi_xx_z + Jy * chi_yy_z \
            + Gamma * chi_xy_z + Gamma * chi_yx_z

        Hamiltonian[0, 4] += -0.25 * 1j * (term_x * expkx + term_y * expky + term_z)

        # b-Majorana hopping H^b (diagonal b^s_A-b^s_B blocks)
        xx = (Kx + Jx) * chi_00_x + Jx * chi_00_y * expky * np.conjugate(expkx) + Jx * chi_00_z * np.conjugate(expkx)
        yy = (Ky + Jy) * chi_00_y + Jy * chi_00_x * expkx * np.conjugate(expky) + Jy * chi_00_z * np.conjugate(expky)
        zz = (Kz + Jz) * chi_00_z + Jz * chi_00_x * expkx + Jz * chi_00_y * expky

        Hamiltonian[1, 5] += -0.25 * 1j * xx  # x-x
        Hamiltonian[2, 6] += -0.25 * 1j * yy  # y-y
        Hamiltonian[3, 7] += -0.25 * 1j * zz  # z-z

        # off-diagonal component: Gamma * eta_gamma bilinears
        # x-bond
        Hamiltonian[2, 7] += -0.25 * 1j * chi_00_x * Gamma * expkx  # y-z
        Hamiltonian[3, 6] += -0.25 * 1j * chi_00_x * Gamma * expkx * np.conjugate(expky)  # z-y
        # y-bond
        Hamiltonian[1, 7] += -0.25 * 1j * chi_00_y * Gamma * expky  # x-z
        Hamiltonian[3, 5] += -0.25 * 1j * chi_00_y * Gamma * expky * np.conjugate(expkx)  # z-x
        # z-bond
        Hamiltonian[1, 6] += -0.25 * 1j * chi_00_z * Gamma * np.conjugate(expky)  # x-y
        Hamiltonian[2, 5] += -0.25 * 1j * chi_00_z * Gamma * np.conjugate(expkx)  # y-x

        # on-site b-c Weiss field H^bc: A-site entries use m_B, B-site entries use m_A

        # x-bond (gamma = x)
        # c bx (beta = x = gamma)
        Hamiltonian[0, 1] += 0.25 * 1j * (Jx + Kx) * mx_B  # A-sublattice J, K; x-direction
        Hamiltonian[4, 5] += 0.25 * 1j * (Jx + Kx) * mx_A * np.conjugate(expkx)  # B-sublattice J, K; x-direction
        # c by: A site b^y_A pairs with m^z_B via Gamma; B site b^y_B pairs with m^z_A via Gamma
        Hamiltonian[0, 2] += 0.25 * 1j * (Jy * my_B + Gamma * mz_B)
        Hamiltonian[4, 6] += 0.25 * 1j * (Jy * my_A + Gamma * mz_A) * np.conjugate(expky)
        # c bz: A site b^z_A <- Gamma m^y_B; B site b^z_B <- Gamma m^y_A
        Hamiltonian[0, 3] += 0.25 * 1j * (Jz * mz_B + Gamma * my_B)
        Hamiltonian[4, 7] += 0.25 * 1j * (Jz * mz_A + Gamma * my_A)

        # y-bond (gamma = y)
        # c bx: A site b^x_A <- Gamma m^z_B; B site b^x_B <- Gamma m^z_A
        Hamiltonian[0, 1] += 0.25 * 1j * (Jx * mx_B + Gamma * mz_B)  # A-sublattice J, K; x-direction
        Hamiltonian[4, 5] += 0.25 * 1j * (Jx * mx_A + Gamma * mz_A) * np.conjugate(expkx)  # B-sublattice J, K; x-direction
        # c by (beta = y = gamma)
        Hamiltonian[0, 2] += 0.25 * 1j * (Jy + Ky) * my_B
        Hamiltonian[4, 6] += 0.25 * 1j * (Jy + Ky) * my_A * np.conjugate(expky)
        # c bz: A site b^z_A <- Gamma m^x_B; B site b^z_B <- Gamma m^x_A
        Hamiltonian[0, 3] += 0.25 * 1j * (Jz * mz_B + Gamma * mx_B)
        Hamiltonian[4, 7] += 0.25 * 1j * (Jz * mz_A + Gamma * mx_A)

        # z-bond (gamma = z)
        # c bx: A site b^x_A <- Gamma m^y_B; B site b^x_B <- Gamma m^y_A
        Hamiltonian[0, 1] += 0.25 * 1j * (Jx * mx_B + Gamma * my_B)  # A-sublattice J, K; x-direction
        Hamiltonian[4, 5] += 0.25 * 1j * (Jx * mx_A + Gamma * my_A) * np.conjugate(expkx)  # B-sublattice J, K; x-direction
        # c by: A site b^y_A <- Gamma m^x_B; B site b^y_B <- Gamma m^x_A
        Hamiltonian[0, 2] += 0.25 * 1j * (Jy * my_B + Gamma * mx_B)
        Hamiltonian[4, 6] += 0.25 * 1j * (Jy * my_A + Gamma * mx_A) * np.conjugate(expky)
        # c bz (beta = z = gamma)
        Hamiltonian[0, 3] += 0.25 * 1j * (Jz + Kz) * mz_B
        Hamiltonian[4, 7] += 0.25 * 1j * (Jz + Kz) * mz_A


        # constraint (Lagrange multipliers)
        Hamiltonian[0, 1] += 0.5 * 1j * lagrange_x
        Hamiltonian[2, 3] += 0.5 * 1j * lagrange_x
        # B-sublattice
        Hamiltonian[4, 5] += 0.5 * 1j * lagrange_x * np.conjugate(expkx)
        Hamiltonian[6, 7] += 0.5 * 1j * lagrange_x * expky

        # lambda_y
        # A-sublattice
        Hamiltonian[0, 2] += 0.5 * 1j * lagrange_y
        Hamiltonian[3, 1] += 0.5 * 1j * lagrange_y
        # B-sublattice
        Hamiltonian[4, 6] += 0.5 * 1j * lagrange_y * np.conjugate(expky)
        Hamiltonian[7, 5] += 0.5 * 1j * lagrange_y * np.conjugate(expkx)

        # lambda_z
        # A-sublattice
        Hamiltonian[0, 3] += 0.5 * 1j * lagrange_z
        Hamiltonian[1, 2] += 0.5 * 1j * lagrange_z
        # B-sublattice
        Hamiltonian[4, 7] += 0.5 * 1j * lagrange_z
        Hamiltonian[5, 6] += 0.5 * 1j * lagrange_z * expkx * np.conjugate(expky)

        Hamiltonian += Hamiltonian.conj().T

        return 2 * Hamiltonian


    def MF_conditions_general(self, MF_parameters):
        """27 self-consistency residuals (computed correlator - parameter; summed constraint for lambda), vectorized over k."""
        assert len(MF_parameters) == 27

        (chi_00_x, chi_00_y, chi_00_z,
         chi_xx_x, chi_yy_x, chi_zz_x,
         chi_xx_y, chi_yy_y, chi_zz_y,
         chi_xx_z, chi_yy_z, chi_zz_z,
         chi_yz_x, chi_zy_x,
         chi_xz_y, chi_zx_y,
         chi_xy_z, chi_yx_z,
         mx_A, my_A, mz_A, mx_B, my_B, mz_B,
         lagrange_x, lagrange_y, lagrange_z) = MF_parameters

        Kx, Ky, Kz = self.Kxyz
        Jx, Jy, Jz = self.Jxyz
        Gamma = self.Gamma

        # (Nx, Ny)-shaped phase grids -- the batched analogue of momentum_function()'s
        # per-k-point expkx/expky (expkz is always 1 and never appears below).
        kx_idx = np.arange(self.Nx).reshape(-1, 1)
        ky_idx = np.arange(self.Ny).reshape(1, -1)
        expkx = np.broadcast_to(np.exp(1j * self.kx_basis * (kx_idx + self.kx_shift)), (self.Nx, self.Ny)).astype(self.dtype)
        expky = np.broadcast_to(np.exp(1j * self.ky_basis * (ky_idx + self.ky_shift)), (self.Nx, self.Ny)).astype(self.dtype)

        # batched Hamiltonian (Nx, Ny, 8, 8): same entries as _get_Hamiltonian_general
        H = np.zeros((self.Nx, self.Ny, 8, 8), dtype=self.dtype)

        term_x = (Kx + Jx) * chi_xx_x + Jy * chi_yy_x + Jz * chi_zz_x \
            + Gamma * chi_yz_x + Gamma * chi_zy_x
        term_y = (Ky + Jy) * chi_yy_y + Jx * chi_xx_y + Jz * chi_zz_y \
            + Gamma * chi_xz_y + Gamma * chi_zx_y
        term_z = (Kz + Jz) * chi_zz_z + Jx * chi_xx_z + Jy * chi_yy_z \
            + Gamma * chi_xy_z + Gamma * chi_yx_z

        H[:, :, 0, 4] += -0.25j * (term_x * expkx + term_y * expky + term_z)

        xx = (Kx + Jx) * chi_00_x + Jx * chi_00_y * expky * np.conjugate(expkx) + Jx * chi_00_z * np.conjugate(expkx)
        yy = (Ky + Jy) * chi_00_y + Jy * chi_00_x * expkx * np.conjugate(expky) + Jy * chi_00_z * np.conjugate(expky)
        zz = (Kz + Jz) * chi_00_z + Jz * chi_00_x * expkx + Jz * chi_00_y * expky

        H[:, :, 1, 5] += -0.25j * xx
        H[:, :, 2, 6] += -0.25j * yy
        H[:, :, 3, 7] += -0.25j * zz

        H[:, :, 2, 7] += -0.25j * chi_00_x * Gamma * expkx
        H[:, :, 3, 6] += -0.25j * chi_00_x * Gamma * expkx * np.conjugate(expky)
        H[:, :, 1, 7] += -0.25j * chi_00_y * Gamma * expky
        H[:, :, 3, 5] += -0.25j * chi_00_y * Gamma * expky * np.conjugate(expkx)
        H[:, :, 1, 6] += -0.25j * chi_00_z * Gamma * np.conjugate(expky)
        H[:, :, 2, 5] += -0.25j * chi_00_z * Gamma * np.conjugate(expkx)

        # x-bond (gamma = x)
        H[:, :, 0, 1] += 0.25j * (Jx + Kx) * mx_B
        H[:, :, 4, 5] += 0.25j * (Jx + Kx) * mx_A * np.conjugate(expkx)
        H[:, :, 0, 2] += 0.25j * (Jy * my_B + Gamma * mz_B)
        H[:, :, 4, 6] += 0.25j * (Jy * my_A + Gamma * mz_A) * np.conjugate(expky)
        H[:, :, 0, 3] += 0.25j * (Jz * mz_B + Gamma * my_B)
        H[:, :, 4, 7] += 0.25j * (Jz * mz_A + Gamma * my_A)

        # y-bond (gamma = y)
        H[:, :, 0, 1] += 0.25j * (Jx * mx_B + Gamma * mz_B)
        H[:, :, 4, 5] += 0.25j * (Jx * mx_A + Gamma * mz_A) * np.conjugate(expkx)
        H[:, :, 0, 2] += 0.25j * (Jy + Ky) * my_B
        H[:, :, 4, 6] += 0.25j * (Jy + Ky) * my_A * np.conjugate(expky)
        H[:, :, 0, 3] += 0.25j * (Jz * mz_B + Gamma * mx_B)
        H[:, :, 4, 7] += 0.25j * (Jz * mz_A + Gamma * mx_A)

        # z-bond (gamma = z)
        H[:, :, 0, 1] += 0.25j * (Jx * mx_B + Gamma * my_B)
        H[:, :, 4, 5] += 0.25j * (Jx * mx_A + Gamma * my_A) * np.conjugate(expkx)
        H[:, :, 0, 2] += 0.25j * (Jy * my_B + Gamma * mx_B)
        H[:, :, 4, 6] += 0.25j * (Jy * my_A + Gamma * mx_A) * np.conjugate(expky)
        H[:, :, 0, 3] += 0.25j * (Jz + Kz) * mz_B
        H[:, :, 4, 7] += 0.25j * (Jz + Kz) * mz_A


        # constraint (Lagrange multipliers)
        H[:, :, 0, 1] += 0.5j * lagrange_x
        H[:, :, 2, 3] += 0.5j * lagrange_x
        H[:, :, 4, 5] += 0.5j * lagrange_x * np.conjugate(expkx)
        H[:, :, 6, 7] += 0.5j * lagrange_x * expky

        H[:, :, 0, 2] += 0.5j * lagrange_y
        H[:, :, 3, 1] += 0.5j * lagrange_y
        H[:, :, 4, 6] += 0.5j * lagrange_y * np.conjugate(expky)
        H[:, :, 7, 5] += 0.5j * lagrange_y * np.conjugate(expkx)

        H[:, :, 0, 3] += 0.5j * lagrange_z
        H[:, :, 1, 2] += 0.5j * lagrange_z
        H[:, :, 4, 7] += 0.5j * lagrange_z
        H[:, :, 5, 6] += 0.5j * lagrange_z * expkx * np.conjugate(expky)

        # Hermitize (batched analogue of Hamiltonian += Hamiltonian.conj().T -- swap only the
        # last two axes, not the batch axes) and apply the same overall factor of 2.
        H = H + np.conj(np.swapaxes(H, -1, -2))
        H = 2 * H

        # one batched diagonalization for the whole k-grid; keep the 4 positive-energy modes
        _, evecs = np.linalg.eigh(H)
        ev = evecs[:, :, :, 4:8]  # (Nx, Ny, 8, 4): positive-energy branch, all 8 Nambu rows
        N = self.Nx * self.Ny

        def _corr(i, j, phase=1.0):
            # Vectorized analogue of the scalar loop's
            # `sum_{kx,ky} np.dot(eigenvector[i], eigenvector[j].conjugate().T) * phase / N`.
            return np.sum(np.sum(ev[:, :, i, :] * np.conj(ev[:, :, j, :]), axis=-1) * phase) / N

        correlator_chi_00_x = 2j * _corr(0, 4, np.conjugate(expkx))
        correlator_chi_00_y = 2j * _corr(0, 4, np.conjugate(expky))
        correlator_chi_00_z = 2j * _corr(0, 4, 1.0)

        correlator_chi_xx_x = 2j * _corr(1, 5, 1.0)
        correlator_chi_xx_y = 2j * _corr(1, 5, np.conjugate(expky) * expkx)
        correlator_chi_xx_z = 2j * _corr(1, 5, expkx)

        correlator_chi_yy_x = 2j * _corr(2, 6, np.conjugate(expkx) * expky)
        correlator_chi_yy_y = 2j * _corr(2, 6, 1.0)
        correlator_chi_yy_z = 2j * _corr(2, 6, expky)

        correlator_chi_zz_x = 2j * _corr(3, 7, np.conjugate(expkx))
        correlator_chi_zz_y = 2j * _corr(3, 7, np.conjugate(expky))
        correlator_chi_zz_z = 2j * _corr(3, 7, 1.0)

        correlator_chi_yz_x = 2j * _corr(2, 7, np.conjugate(expkx))
        correlator_chi_zy_x = 2j * _corr(3, 6, np.conjugate(expkx) * expky)
        correlator_chi_xz_y = 2j * _corr(1, 7, np.conjugate(expky))
        correlator_chi_zx_y = 2j * _corr(3, 5, np.conjugate(expky) * expkx)
        correlator_chi_xy_z = 2j * _corr(1, 6, expky)
        correlator_chi_yx_z = 2j * _corr(2, 5, expkx)

        correlator_mx_A = 2j * _corr(0, 1, 1.0)
        correlator_mx_B = 2j * _corr(4, 5, expkx)
        correlator_my_A = 2j * _corr(0, 2, 1.0)
        correlator_my_B = 2j * _corr(4, 6, expky)
        correlator_mz_A = 2j * _corr(0, 3, 1.0)
        correlator_mz_B = 2j * _corr(4, 7, 1.0)

        correlator_lagrange_x = 2j * (_corr(0, 1, 1.0) + _corr(2, 3, 1.0)
                                       + _corr(4, 5, expkx) + _corr(6, 7, np.conjugate(expky)))
        correlator_lagrange_y = 2j * (_corr(0, 2, 1.0) + _corr(3, 1, 1.0)
                                       + _corr(4, 6, expky) + _corr(7, 5, expkx))
        correlator_lagrange_z = 2j * (_corr(0, 3, 1.0) + _corr(1, 2, 1.0)
                                       + _corr(4, 7, 1.0) + _corr(5, 6, np.conjugate(expkx) * expky))

        return (
            np.real(correlator_chi_00_x - chi_00_x),
            np.real(correlator_chi_00_y - chi_00_y),
            np.real(correlator_chi_00_z - chi_00_z),
            np.real(correlator_chi_xx_x - chi_xx_x),
            np.real(correlator_chi_yy_x - chi_yy_x),
            np.real(correlator_chi_zz_x - chi_zz_x),
            np.real(correlator_chi_xx_y - chi_xx_y),
            np.real(correlator_chi_yy_y - chi_yy_y),
            np.real(correlator_chi_zz_y - chi_zz_y),
            np.real(correlator_chi_xx_z - chi_xx_z),
            np.real(correlator_chi_yy_z - chi_yy_z),
            np.real(correlator_chi_zz_z - chi_zz_z),
            np.real(correlator_chi_yz_x - chi_yz_x),
            np.real(correlator_chi_zy_x - chi_zy_x),
            np.real(correlator_chi_xz_y - chi_xz_y),
            np.real(correlator_chi_zx_y - chi_zx_y),
            np.real(correlator_chi_xy_z - chi_xy_z),
            np.real(correlator_chi_yx_z - chi_yx_z),
            np.real(correlator_mx_A - mx_A),
            np.real(correlator_my_A - my_A),
            np.real(correlator_mz_A - mz_A),
            np.real(correlator_mx_B - mx_B),
            np.real(correlator_my_B - my_B),
            np.real(correlator_mz_B - mz_B),
            np.real(correlator_lagrange_x),
            np.real(correlator_lagrange_y),
            np.real(correlator_lagrange_z),
        )

    def _MF_conditions_general_scalar_reference(self, MF_parameters):
        """Per-k-point (slow) version of MF_conditions_general, kept as a cross-check."""
        assert len(MF_parameters) == 27

        (chi_00_x, chi_00_y, chi_00_z,
         chi_xx_x, chi_yy_x, chi_zz_x,
         chi_xx_y, chi_yy_y, chi_zz_y,
         chi_xx_z, chi_yy_z, chi_zz_z,
         chi_yz_x, chi_zy_x,
         chi_xz_y, chi_zx_y,
         chi_xy_z, chi_yx_z,
         mx_A, my_A, mz_A, mx_B, my_B, mz_B,
         lagrange_x, lagrange_y, lagrange_z) = MF_parameters

        correlator_chi_00_x = 0
        correlator_chi_00_y = 0
        correlator_chi_00_z = 0

        correlator_chi_xx_x = 0
        correlator_chi_yy_x = 0
        correlator_chi_zz_x = 0
        correlator_chi_xx_y = 0
        correlator_chi_yy_y = 0
        correlator_chi_zz_y = 0
        correlator_chi_xx_z = 0
        correlator_chi_yy_z = 0
        correlator_chi_zz_z = 0

        correlator_chi_yz_x = 0
        correlator_chi_zy_x = 0
        correlator_chi_xz_y = 0
        correlator_chi_zx_y = 0
        correlator_chi_xy_z = 0
        correlator_chi_yx_z = 0

        correlator_mx_A = 0
        correlator_my_A = 0
        correlator_mz_A = 0
        correlator_mx_B = 0
        correlator_my_B = 0
        correlator_mz_B = 0

        correlator_lagrange_x = 0
        correlator_lagrange_y = 0
        correlator_lagrange_z = 0

        for kx in np.arange(self.Nx):
            for ky in np.arange(self.Ny):
                expkx, expky, expkz = self.momentum_function(kx, ky)

                eigenvector = LA.eigh(self.get_Hamiltonian(kx, ky, MF_parameters))[1][:, 4:8]
                N = self.Nx * self.Ny

                # eta_gamma = i<c_A c_B> on bond gamma (species c has no gauge shift, so the
                # extraction phase is simply conj(expk_gamma))
                correlator_chi_00_x += 2j * np.dot(eigenvector[0], eigenvector[4].conjugate().T) * np.conjugate(expkx) / N
                correlator_chi_00_y += 2j * np.dot(eigenvector[0], eigenvector[4].conjugate().T) * np.conjugate(expky) / N
                correlator_chi_00_z += 2j * np.dot(eigenvector[0], eigenvector[4].conjugate().T) / N

                # diagonal Q_gamma^{ss}: extraction phase conj(expk_gamma) * expk_s
                correlator_chi_xx_x += 2j * np.dot(eigenvector[1], eigenvector[5].conjugate().T) / N
                correlator_chi_xx_y += 2j * np.dot(eigenvector[1], eigenvector[5].conjugate().T) * np.conjugate(expky) * expkx / N
                correlator_chi_xx_z += 2j * np.dot(eigenvector[1], eigenvector[5].conjugate().T) * expkx / N

                correlator_chi_yy_x += 2j * np.dot(eigenvector[2], eigenvector[6].conjugate().T) * np.conjugate(expkx) * expky / N
                correlator_chi_yy_y += 2j * np.dot(eigenvector[2], eigenvector[6].conjugate().T) / N
                correlator_chi_yy_z += 2j * np.dot(eigenvector[2], eigenvector[6].conjugate().T) * expky / N

                correlator_chi_zz_x += 2j * np.dot(eigenvector[3], eigenvector[7].conjugate().T) * np.conjugate(expkx) / N
                correlator_chi_zz_y += 2j * np.dot(eigenvector[3], eigenvector[7].conjugate().T) * np.conjugate(expky) / N
                correlator_chi_zz_z += 2j * np.dot(eigenvector[3], eigenvector[7].conjugate().T) / N

                # off-diagonal Q_gamma^{st} (s != t): extraction phase conj(expk_gamma) * expk_t
                correlator_chi_yz_x += 2j * np.dot(eigenvector[2], eigenvector[7].conjugate().T) * expkx.conj() / N
                correlator_chi_zy_x += 2j * np.dot(eigenvector[3], eigenvector[6].conjugate().T) * np.conjugate(expkx) * expky / N
                correlator_chi_xz_y += 2j * np.dot(eigenvector[1], eigenvector[7].conjugate().T) * expky.conj() / N
                correlator_chi_zx_y += 2j * np.dot(eigenvector[3], eigenvector[5].conjugate().T) * np.conjugate(expky) * expkx / N
                correlator_chi_xy_z += 2j * np.dot(eigenvector[1], eigenvector[6].conjugate().T) * expky / N
                correlator_chi_yx_z += 2j * np.dot(eigenvector[2], eigenvector[5].conjugate().T) * expkx / N

                # sublattice magnetizations m_A, m_B
                correlator_mx_A += 2j * np.dot(eigenvector[0], eigenvector[1].conjugate().T) / N
                correlator_mx_B += 2j * np.dot(eigenvector[4], eigenvector[5].conjugate().T) * expkx / N
                correlator_my_A += 2j * np.dot(eigenvector[0], eigenvector[2].conjugate().T) / N
                correlator_my_B += 2j * np.dot(eigenvector[4], eigenvector[6].conjugate().T) * expky / N
                correlator_mz_A += 2j * np.dot(eigenvector[0], eigenvector[3].conjugate().T) / N
                correlator_mz_B += 2j * np.dot(eigenvector[4], eigenvector[7].conjugate().T) / N

                # Lagrange multiplier (site constraint, not bond-dependent)
                correlator_lagrange_x += \
                    2j * (np.dot(eigenvector[0], eigenvector[1].conjugate().T) + np.dot(eigenvector[2], eigenvector[
                        3].conjugate().T) + np.dot(eigenvector[4], eigenvector[5].conjugate().T) * expkx + np.dot(
                        eigenvector[6], eigenvector[7].conjugate().T) * np.conjugate(expky)) / N
                correlator_lagrange_y += \
                    2j * (np.dot(eigenvector[0], eigenvector[2].conjugate().T) + np.dot(eigenvector[3], eigenvector[
                        1].conjugate().T) + np.dot(eigenvector[4], eigenvector[6].conjugate().T) * expky + np.dot(
                        eigenvector[7], eigenvector[5].conjugate().T) * expkx) / N
                correlator_lagrange_z += \
                    2j * (np.dot(eigenvector[0], eigenvector[3].conjugate().T) + np.dot(eigenvector[1], eigenvector[
                        2].conjugate().T) + np.dot(eigenvector[4], eigenvector[7].conjugate().T) + np.dot(
                        eigenvector[5], eigenvector[6].conjugate().T) * np.conjugate(expkx) * expky) / N

        return (
            np.real(correlator_chi_00_x - chi_00_x),
            np.real(correlator_chi_00_y - chi_00_y),
            np.real(correlator_chi_00_z - chi_00_z),
            np.real(correlator_chi_xx_x - chi_xx_x),
            np.real(correlator_chi_yy_x - chi_yy_x),
            np.real(correlator_chi_zz_x - chi_zz_x),
            np.real(correlator_chi_xx_y - chi_xx_y),
            np.real(correlator_chi_yy_y - chi_yy_y),
            np.real(correlator_chi_zz_y - chi_zz_y),
            np.real(correlator_chi_xx_z - chi_xx_z),
            np.real(correlator_chi_yy_z - chi_yy_z),
            np.real(correlator_chi_zz_z - chi_zz_z),
            np.real(correlator_chi_yz_x - chi_yz_x),
            np.real(correlator_chi_zy_x - chi_zy_x),
            np.real(correlator_chi_xz_y - chi_xz_y),
            np.real(correlator_chi_zx_y - chi_zx_y),
            np.real(correlator_chi_xy_z - chi_xy_z),
            np.real(correlator_chi_yx_z - chi_yx_z),
            np.real(correlator_mx_A - mx_A),
            np.real(correlator_my_A - my_A),
            np.real(correlator_mz_A - mz_A),
            np.real(correlator_mx_B - mx_B),
            np.real(correlator_my_B - my_B),
            np.real(correlator_mz_B - mz_B),
            np.real(correlator_lagrange_x),
            np.real(correlator_lagrange_y),
            np.real(correlator_lagrange_z),
        )

    # XXZ phase-diagram machinery: seeding, branch competition, classification.

    def _set_Jxyz(self, Jxyz):
        """Change Jxyz in place without touching MF_parameters (for warm-started sweeps)."""
        self.Jxyz = np.asarray(Jxyz, dtype=float)

    def majorana_gap(self):
        """Lowest positive Majorana band energy over the k-grid at the current solution."""
        Ek, _ = self.get_eiegensystem_Majorana()
        return float(np.min(Ek[..., 0]))

    def build_seed(self, kind, m0=0.3, perturbation=0.01, seed=None):
        """Start vector for one seed kind: 'QSL', '{FM,AFM}_{x,y,z,111}' (|m| = m0) or 'random'."""
        base = np.array(self.warm_start_from_pure_Kitaev(perturbation=perturbation, seed=seed))
        rng = np.random.default_rng(None if seed is None else seed + 1)

        if kind == 'QSL':
            pass  # base's own small random perturbation (from warm_start_from_pure_Kitaev)
                  # is already exactly the right kind of seed here.
        elif kind == 'random':
            base[18:24] = m0 * rng.standard_normal(6)
        else:
            try:
                prefix, axis = kind.split('_', 1)
            except ValueError:
                raise NotImplementedError(
                    "build_seed: unrecognized kind %r (expected 'QSL', 'random', or "
                    "'{FM,AFM}_{x,y,z,111}')" % kind)
            if prefix not in ('FM', 'AFM') or axis not in ('x', 'y', 'z', '111'):
                raise NotImplementedError(
                    "build_seed: unrecognized kind %r (expected 'QSL', 'random', or "
                    "'{FM,AFM}_{x,y,z,111}')" % kind)
            vec = (np.ones(3) / np.sqrt(3)) if axis == '111' else \
                np.array([1.0 if a == axis else 0.0 for a in ('x', 'y', 'z')])
            mA = m0 * vec
            mB = mA if prefix == 'FM' else -mA
            base[18:24] = np.concatenate([mA, mB])
            base[18:24] += perturbation * rng.standard_normal(6)

        return base

    # Default seeds: m = 0 (QSL), FM and AFM along x, y, z, [111].
    DEFAULT_SEED_KINDS = ('QSL', 'FM_x', 'FM_y', 'FM_z', 'FM_111',
                          'AFM_x', 'AFM_y', 'AFM_z', 'AFM_111')

    def classify_solution(self, m_tol=1e-3, collinear_tol=1e-2, bond_sym_tol=1e-3):
        """Classify the current solution: QSL / QSL (bond-nematic!) / FM / AFM / canted/other, plus gap and energy."""
        vals = self.compute_all_MF_parameters()
        mA = np.array([vals['m^x_A'], vals['m^y_A'], vals['m^z_A']])
        mB = np.array([vals['m^x_B'], vals['m^y_B'], vals['m^z_B']])
        magA, magB = float(np.linalg.norm(mA)), float(np.linalg.norm(mB))

        bond_symmetry_broken = abs(vals['chi^00_x'] - vals['chi^00_y']) > bond_sym_tol

        if magA < m_tol and magB < m_tol:
            pattern = 'QSL (bond-nematic!)' if bond_symmetry_broken else 'QSL'
        elif magA < m_tol or magB < m_tol:
            pattern = 'canted/other'  # one sublattice ordered, the other not -- unusual,
                                      # flag it rather than silently calling it FM or AFM.
        else:
            cos_angle = float(np.dot(mA, mB) / (magA * magB))
            if cos_angle > 1 - collinear_tol:
                pattern = 'FM'
            elif cos_angle < -1 + collinear_tol:
                pattern = 'AFM'
            else:
                pattern = 'canted/other'

        gap = self.majorana_gap()

        return {
            'pattern': pattern, '|m_A|': magA, '|m_B|': magB, 'm_A': mA, 'm_B': mB,
            'gap': gap, 'gapless': gap < 1e-3,
            'bond_symmetry_broken': bond_symmetry_broken,
            'chi00_xyz': (vals['chi^00_x'], vals['chi^00_y'], vals['chi^00_z']),
            'variational_energy': self.MF_variational_energy(),
        }

    def solve_best_branch(self, seed_kinds=None, m0=0.3, n_random=2, base_seed=0,
                          e_tol=1e-6, verbose=False):
        """Solve from all seeds, keep converged distinct solutions, return the lowest-energy one and all results."""
        if seed_kinds is None:
            seed_kinds = self.DEFAULT_SEED_KINDS

        results = []
        for kind in seed_kinds:
            self.MF_parameters = self.build_seed(kind, m0=m0, seed=base_seed)
            try:
                self.Self_Consistent_Solution()
            except Exception as exc:
                if verbose:
                    print("  seed %-10r -> solver raised (%r), skipping" % (kind, exc))
                continue
            info = self.classify_solution()
            info['seed'] = kind
            info['MF_parameters'] = np.array(self.MF_parameters)
            results.append(info)
            if verbose:
                print("  [%s] seed %-10r -> %-12s E_MF=%+.8f  gap=%.4f  |m_A|=%.4f  |m_B|=%.4f"
                      % (time.strftime('%H:%M:%S'), kind, info['pattern'],
                         info['variational_energy'], info['gap'],
                         info['|m_A|'], info['|m_B|']))

        for i in range(n_random):
            self.MF_parameters = self.build_seed('random', m0=m0, seed=base_seed + 100 + i)
            try:
                self.Self_Consistent_Solution()
            except Exception as exc:
                if verbose:
                    print("  seed 'random_%d' -> solver raised (%r), skipping" % (i, exc))
                continue
            info = self.classify_solution()
            info['seed'] = 'random_%d' % i
            info['MF_parameters'] = np.array(self.MF_parameters)
            results.append(info)
            if verbose:
                print("  [%s] seed 'random_%d' -> %-12s E_MF=%+.8f  gap=%.4f"
                      % (time.strftime('%H:%M:%S'), i, info['pattern'],
                         info['variational_energy'], info['gap']))

        # keep only seeds with max|residual| <= 1e-6; if none, keep the best one marked unconverged
        RESIDUAL_TOL = 1e-6
        for info in results:
            info['residual'] = float(np.max(np.abs(self.MF_conditions(info['MF_parameters']))))
        converged = [info for info in results if info['residual'] <= RESIDUAL_TOL]
        if verbose:
            for info in results:
                if info['residual'] > RESIDUAL_TOL:
                    print("  seed '%s' -> NOT converged (residual %.1e), discarded"
                          % (info['seed'], info['residual']))
        if converged:
            for info in converged:
                info['unconverged'] = False
            results = converged
        elif results:
            best = min(results, key=lambda r: r['residual'])
            best['unconverged'] = True
            print('  WARNING solve_best_branch: no seed converged (best residual %.1e) at '
                  'Kxyz=%s, Jxyz=%s, Gamma=%s -- keeping it, marked unconverged.'
                  % (best['residual'], list(self.Kxyz), list(self.Jxyz), self.Gamma))
            results = [best]

        if not results:
            raise RuntimeError(
                'solve_best_branch: every seed failed to converge at Kxyz=%s, Jxyz=%s, '
                'Gamma=%s.' % (list(self.Kxyz), list(self.Jxyz), self.Gamma))

        # merge seeds that converged to the same solution (compare the 24 physical parameters, not lambda)
        _N_LAGRANGE = 3  # trailing lagrange_x, lagrange_y, lagrange_z -- see MF_parameters'
                          # unpacking order used throughout this file (e.g. GS_Energy()).

        def _same_fixed_point(r1, r2, atol=1e-5, e_atol=1e-8, energy_gated_atol=1e-2):
            p1 = np.asarray(r1['MF_parameters'])
            p2 = np.asarray(r2['MF_parameters'])
            n_phys = len(p1) - _N_LAGRANGE if len(p1) > _N_LAGRANGE else len(p1)
            if np.allclose(p1[:n_phys], p2[:n_phys], atol=atol):
                return True
            # if energies agree to e_atol, accept a looser parameter match
            if abs(r1['variational_energy'] - r2['variational_energy']) < e_atol:
                return np.allclose(p1[:n_phys], p2[:n_phys], atol=energy_gated_atol)
            return False

        distinct = []
        for r in results:
            if not any(_same_fixed_point(r, d) for d in distinct):
                distinct.append(r)

        winner_obj = min(distinct, key=lambda r: r['variational_energy'])
        best = dict(winner_obj)  # don't mutate the copy stored in `results`/`distinct`
        best['near_degenerate_with'] = [
            # other distinct branches within e_tol of the winner
            r['seed'] for r in distinct
            if r is not winner_obj and abs(r['variational_energy'] - best['variational_energy']) < e_tol]
        # energy gap to the second-lowest distinct branch
        sorted_E = sorted(r['variational_energy'] for r in distinct)
        best['energy_margin'] = (sorted_E[1] - sorted_E[0]) if len(sorted_E) > 1 else float('inf')

        if best['bond_symmetry_broken']:
            print(
                "  *** WARNING: winning branch at Kxyz=%s, Jxyz=%s, Gamma=%s spontaneously "
                "breaks the x<->y bond symmetry the Hamiltonian itself has (chi^00_x=%.4f "
                "vs chi^00_y=%.4f) -- see classify_solution()'s docstring: this has been "
                "OBSERVED to be Nx-dependent (a likely finite-size artifact of the "
                "self-consistency root-find, not confirmed physics). Re-run this point at a "
                "larger Nx/Ny before trusting it, and do not report this branch as an "
                "ordinary QSL without that check. ***"
                % (list(self.Kxyz), list(self.Jxyz), self.Gamma,
                   best['chi00_xyz'][0], best['chi00_xyz'][1]))

        self.MF_parameters = best['MF_parameters']
        return best, results


# Phase-diagram drivers. The 2-site cell describes QSL, FM, AFM (Neel) and canted states, not zigzag/stripy.

DEFAULT_SEED_KINDS = Majorana_rep.DEFAULT_SEED_KINDS


def continuation_scan(points, Nx=18, Ny=18, seed_kinds=None, m0=0.3,
                      base_seed=0, reseed_every=1, verbose=True):
    """Scan a 1D path of (K, J_perp, Gamma, Delta) points (Jz = J_perp*Delta); multi-seed solve at each point."""
    if seed_kinds is None:
        seed_kinds = Majorana_rep.DEFAULT_SEED_KINDS

    tracked = {}  # seed label -> most recently converged 27-vector, for warm-starting
    out = []

    for i, (K, J_perp, Gamma, delta) in enumerate(points):
        Kxyz = np.array([K, K, K])
        Jxyz = np.array([J_perp, J_perp, J_perp * delta])
        parameters = Parameters(Nx, Ny, Kxyz, Jxyz, Gamma)
        model = Majorana_rep(parameters)

        do_full_reseed = (i % reseed_every == 0) or not tracked
        point_results = []

        if do_full_reseed:
            best, all_results = model.solve_best_branch(
                seed_kinds=seed_kinds, m0=m0, base_seed=base_seed, verbose=False)
            point_results = all_results
            for r in all_results:
                tracked[r['seed']] = r['MF_parameters']
        else:
            for kind, warm_params in list(tracked.items()):
                model.MF_parameters = np.array(warm_params)
                try:
                    model.Self_Consistent_Solution()
                except Exception:
                    continue
                info = model.classify_solution()
                info['seed'] = kind
                info['MF_parameters'] = np.array(model.MF_parameters)
                point_results.append(info)
                tracked[kind] = info['MF_parameters']

        if not point_results:
            raise RuntimeError(
                'continuation_scan: every branch failed at point %d '
                '(K=%s, J_perp=%s, Gamma=%s, Delta=%s).' % (i, K, J_perp, Gamma, delta))

        winner = min(point_results, key=lambda r: r['variational_energy'])
        if 'energy_margin' not in winner:
            # energy margin for warm-start-only points
            sorted_E = sorted(r['variational_energy'] for r in point_results)
            winner['energy_margin'] = (sorted_E[1] - sorted_E[0]) if len(sorted_E) > 1 else float('inf')

        if verbose:
            print('[%3d] K=%+.3f J_perp=%+.4f Gamma=%+.4f Delta=%+.3f  ->  %-12s '
                  '(seed %r)  E=%+.8f  gap=%.4f'
                  % (i, K, J_perp, Gamma, delta, winner['pattern'], winner['seed'],
                     winner['variational_energy'], winner['gap']))

        out.append({'K': K, 'J_perp': J_perp, 'Gamma': Gamma, 'delta': delta,
                    'Nx': Nx, 'Ny': Ny,
                    'branches': point_results, 'winner': winner})

    return out


# Order of the 27 MF parameters (also the column order of the text tables).
MF_PARAMETER_NAMES = (
    'chi_00_x', 'chi_00_y', 'chi_00_z',
    'chi_xx_x', 'chi_yy_x', 'chi_zz_x',
    'chi_xx_y', 'chi_yy_y', 'chi_zz_y',
    'chi_xx_z', 'chi_yy_z', 'chi_zz_z',
    'chi_yz_x', 'chi_zy_x',
    'chi_xz_y', 'chi_zx_y',
    'chi_xy_z', 'chi_yx_z',
    'mx_A', 'my_A', 'mz_A', 'mx_B', 'my_B', 'mz_B',
    'lagrange_x', 'lagrange_y', 'lagrange_z',
)
assert len(MF_PARAMETER_NAMES) == 27


def _JperpJz_text_header():
    """Header line (tab-separated) for the plain-text (J_perp/|K|, J_z/|K|) export -- see
    _append_JperpJz_point_to_text()'s docstring for what each column means."""
    return '\t'.join(['K', 'J_perp', 'Gamma', 'Jz', 'Nx', 'Ny', 'pattern', 'variational_energy']
                      + list(MF_PARAMETER_NAMES)) + '\n'


def _append_JperpJz_point_to_text(text_path, point):
    """Append one point (couplings, winning phase, energy, 27 MF parameters) to the text table."""
    w = point['winner']
    mf = np.asarray(w['MF_parameters'], dtype=float).ravel()
    if mf.size != 27:
        raise ValueError('_append_JperpJz_point_to_text: expected 27 MF parameters, got %d'
                          % mf.size)
    fields = [repr(float(point['K'])), repr(float(point['J_perp'])),
              repr(float(point['Gamma'])), repr(float(point['Jz'])),
              str(point['Nx']), str(point['Ny']),
              w['pattern'], repr(float(w['variational_energy']))]
    fields += [repr(float(x)) for x in mf]
    # create the output folder if needed
    text_dir = os.path.dirname(text_path)
    if text_dir:
        os.makedirs(text_dir, exist_ok=True)
    write_header = not (os.path.exists(text_path) and os.path.getsize(text_path) > 0)
    with open(text_path, 'a') as f:
        if write_header:
            f.write(_JperpJz_text_header())
        f.write('\t'.join(fields) + '\n')


def export_JperpJz_results_to_text(results_jj, text_path):
    """Write a finished (J_perp, J_z) scan to a fresh text table; returns the number of rows."""
    if os.path.exists(text_path):
        os.remove(text_path)
    j_perps = sorted(results_jj.keys())
    n = 0
    for jp in j_perps:
        for jz in sorted(results_jj[jp].keys()):
            _append_JperpJz_point_to_text(text_path, results_jj[jp][jz])
            n += 1
    return n


def continuation_scan_JperpJz(points, Nx=18, Ny=18, seed_kinds=None, m0=0.3,
                              base_seed=0, reseed_every=1, verbose=True, text_path=None):
    """Scan a 1D path of (K, J_perp, Gamma, Jz) points; multi-seed solve at each point."""
    if seed_kinds is None:
        seed_kinds = Majorana_rep.DEFAULT_SEED_KINDS

    tracked = {}  # seed label -> most recently converged 27-vector, for warm-starting
    out = []

    for i, (K, J_perp, Gamma, Jz) in enumerate(points):
        Kxyz = np.array([K, K, K])
        Jxyz = np.array([J_perp, J_perp, Jz])
        parameters = Parameters(Nx, Ny, Kxyz, Jxyz, Gamma)
        model = Majorana_rep(parameters)

        do_full_reseed = (i % reseed_every == 0) or not tracked
        point_results = []

        if do_full_reseed:
            best, all_results = model.solve_best_branch(
                seed_kinds=seed_kinds, m0=m0, base_seed=base_seed, verbose=False)
            point_results = all_results
            for r in all_results:
                tracked[r['seed']] = r['MF_parameters']
        else:
            for kind, warm_params in list(tracked.items()):
                model.MF_parameters = np.array(warm_params)
                try:
                    model.Self_Consistent_Solution()
                except Exception:
                    continue
                info = model.classify_solution()
                info['seed'] = kind
                info['MF_parameters'] = np.array(model.MF_parameters)
                point_results.append(info)
                tracked[kind] = info['MF_parameters']

        if not point_results:
            raise RuntimeError(
                'continuation_scan_JperpJz: every branch failed at point %d '
                '(K=%s, J_perp=%s, Gamma=%s, Jz=%s).' % (i, K, J_perp, Gamma, Jz))

        winner = min(point_results, key=lambda r: r['variational_energy'])
        if 'energy_margin' not in winner:
            sorted_E = sorted(r['variational_energy'] for r in point_results)
            winner['energy_margin'] = (sorted_E[1] - sorted_E[0]) if len(sorted_E) > 1 else float('inf')

        if verbose:
            print('[%3d] K=%+.3f J_perp=%+.4f Gamma=%+.4f Jz=%+.4f  ->  %-12s '
                  '(seed %r)  E=%+.8f  gap=%.4f'
                  % (i, K, J_perp, Gamma, Jz, winner['pattern'], winner['seed'],
                     winner['variational_energy'], winner['gap']))

        point = {'K': K, 'J_perp': J_perp, 'Gamma': Gamma, 'Jz': Jz,
                 'Nx': Nx, 'Ny': Ny,
                 'branches': point_results, 'winner': winner}
        out.append(point)
        if text_path is not None:
            _append_JperpJz_point_to_text(text_path, point)

    return out


def scan_XXZ_1D(K, J_perp, Gamma, delta_list, **kwargs):
    """1D cut along Delta = Jz/J_perp at fixed K, J_perp, Gamma (wrapper of continuation_scan)."""
    points = [(K, J_perp, Gamma, d) for d in delta_list]
    return continuation_scan(points, **kwargs)


def build_phase_diagram_2D(K, delta_list, J_perp_over_K_list, Gamma_over_K_list, **kwargs):
    """Family of (J_perp, Gamma) phase diagrams, one per Delta."""
    results = {}
    for delta in delta_list:
        results[delta] = {}
        for g in Gamma_over_K_list:
            Gamma = g * K
            points = [(K, jp * K, Gamma, delta) for jp in J_perp_over_K_list]
            row = continuation_scan(points, **kwargs)
            results[delta][g] = {jp: pt for jp, pt in zip(J_perp_over_K_list, row)}
    return results


def build_phase_diagram_JperpDelta(K, J_perp_over_K_list, delta_list, Gamma_over_K=0.0, **kwargs):
    """(J_perp, Delta) phase diagram at fixed Gamma, solved row by row."""
    Gamma = Gamma_over_K * K
    results = {}
    for jp in J_perp_over_K_list:
        J_perp = jp * K
        points = [(K, J_perp, Gamma, d) for d in delta_list]
        row = continuation_scan(points, **kwargs)
        results[jp] = {d: pt for d, pt in zip(delta_list, row)}
    return results


def build_phase_diagram_JperpJz(K, J_perp_over_K_list, Jz_over_K_list, Gamma_over_K=0.0,
                                text_path=None, **kwargs):
    """(J_perp, J_z) phase diagram at fixed Gamma, solved row by row (serial)."""
    Gamma = Gamma_over_K * K
    results = {}
    for jp in J_perp_over_K_list:
        J_perp = jp * K
        points = [(K, J_perp, Gamma, jz * K) for jz in Jz_over_K_list]
        row = continuation_scan_JperpJz(points, text_path=text_path, **kwargs)
        results[jp] = {jz: pt for jz, pt in zip(Jz_over_K_list, row)}
    return results


CHECKPOINT_DIR = os.getcwd() + '/paper_plots/checkpoints'


def _JperpDelta_checkpoint_path(K, J_perp_over_K_list, delta_list, Gamma_over_K, Nx, Ny):
    """Checkpoint file name encoding the (J_perp, Delta) grid, Gamma and Nx, Ny."""
    jp_lo, jp_hi, jp_n = min(J_perp_over_K_list), max(J_perp_over_K_list), len(J_perp_over_K_list)
    d_lo, d_hi, d_n = min(delta_list), max(delta_list), len(delta_list)
    return (f'{CHECKPOINT_DIR}/JperpDelta_K{K:g}_Jperp{jp_lo:g}-{jp_hi:g}x{jp_n}_'
            f'Delta{d_lo:g}-{d_hi:g}x{d_n}_Gamma{Gamma_over_K:g}_Nx{Nx}Ny{Ny}.pkl')


def _JperpJz_checkpoint_path(K, J_perp_over_K_list, Jz_over_K_list, Gamma_over_K, Nx, Ny):
    """Checkpoint file name encoding the (J_perp, J_z) grid, Gamma, Nx, Ny and boundary conditions."""
    jp_lo, jp_hi, jp_n = min(J_perp_over_K_list), max(J_perp_over_K_list), len(J_perp_over_K_list)
    jz_lo, jz_hi, jz_n = min(Jz_over_K_list), max(Jz_over_K_list), len(Jz_over_K_list)
    return (f'{CHECKPOINT_DIR}/JperpJz_K{K:g}_Jperp{jp_lo:g}-{jp_hi:g}x{jp_n}_'
            f'Jz{jz_lo:g}-{jz_hi:g}x{jz_n}_Gamma{Gamma_over_K:g}_Nx{Nx}Ny{Ny}{_bc_tag()}.pkl')


def _bc_tag():
    """'' for the default periodic k-grid; '_BCx<sx>y<sy>' otherwise, so checkpoints from
    different Majorana boundary-condition sectors can never be mixed up."""
    sx, sy = system_parameters.kx_shift, system_parameters.ky_shift
    return '' if (sx == 0 and sy == 0) else '_BCx%gy%g' % (sx, sy)


def _save_checkpoint(path, obj):
    """Pickle `obj` to `path` atomically (temporary file + os.replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


def _load_checkpoint(path, verbose=True):
    """Load a checkpoint written by _save_checkpoint(); None if missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
    except Exception as exc:
        if verbose:
            print('  [checkpoint] could not load %r (%r) -- starting fresh.' % (path, exc))
        return None
    if verbose:
        print('  [checkpoint] found existing checkpoint at %r' % path)
    return obj


def _solve_JperpDelta_row(args):
    """Worker: solve one J_perp row of the (J_perp, Delta) diagram."""
    K, jp, J_perp, delta_list, Gamma, row_kwargs = args
    row_kwargs = dict(row_kwargs)
    verbose = row_kwargs.pop('verbose', True)
    points = [(K, J_perp, Gamma, d) for d in delta_list]
    if verbose:
        print('[pid %d] starting row J_perp/|K|=%+.4f (%d delta points)'
              % (os.getpid(), jp, len(delta_list)), flush=True)
    # no per-point output from parallel workers
    row = continuation_scan(points, verbose=False, **row_kwargs)
    if verbose:
        print('[pid %d] finished row J_perp/|K|=%+.4f' % (os.getpid(), jp), flush=True)
    return jp, row


def build_phase_diagram_JperpDelta_parallel(K, J_perp_over_K_list, delta_list, Gamma_over_K=0.0,
                                            n_workers=None, checkpoint=True, checkpoint_path=None,
                                            resume=True, **kwargs):
    """Parallel (one process per row) version of build_phase_diagram_JperpDelta, with checkpointing."""
    import multiprocessing as mp

    Gamma = Gamma_over_K * K
    verbose = kwargs.pop('verbose', True)
    Nx = kwargs.get('Nx', 18)
    Ny = kwargs.get('Ny', Nx)

    if checkpoint and checkpoint_path is None:
        checkpoint_path = _JperpDelta_checkpoint_path(K, J_perp_over_K_list, delta_list,
                                                       Gamma_over_K, Nx, Ny)

    results = {}
    if checkpoint and resume:
        loaded = _load_checkpoint(checkpoint_path, verbose=verbose)
        if loaded is not None:
            results = loaded
            if verbose:
                print('  [checkpoint] %d/%d row(s) already complete -- only solving the rest'
                      % (len(results), len(J_perp_over_K_list)))

    def _checkpoint_now():
        if checkpoint:
            _save_checkpoint(checkpoint_path, results)

    tasks = [(K, jp, jp * K, delta_list, Gamma, dict(kwargs, verbose=verbose))
             for jp in J_perp_over_K_list if jp not in results]

    if not tasks:
        if verbose:
            print('build_phase_diagram_JperpDelta_parallel: all %d row(s) already complete '
                  '(from checkpoint) -- nothing left to solve.' % len(J_perp_over_K_list))
        return {jp: results[jp] for jp in J_perp_over_K_list}

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(tasks)))

    if verbose:
        print('build_phase_diagram_JperpDelta_parallel: %d row(s) remaining across %d worker '
              'process(es)' % (len(tasks), n_workers))

    if n_workers == 1:
        # No point spinning up a Pool for a single worker -- solve serially in-process
        # instead (also avoids fork/spawn startup overhead).
        for task in tasks:
            jp, row = _solve_JperpDelta_row(task)
            results[jp] = {d: pt for d, pt in zip(delta_list, row)}
            _checkpoint_now()
        return {jp: results[jp] for jp in J_perp_over_K_list}

    with mp.Pool(processes=n_workers) as pool:
        for jp, row in pool.imap_unordered(_solve_JperpDelta_row, tasks):
            results[jp] = {d: pt for d, pt in zip(delta_list, row)}
            _checkpoint_now()
            if verbose:
                print('  [main] %d/%d rows complete (just finished J_perp/|K|=%+.4f)'
                      % (len(results), len(J_perp_over_K_list), jp))

    # restore the original row order
    return {jp: results[jp] for jp in J_perp_over_K_list}


def _solve_JperpJz_row(args):
    """Worker: solve one J_perp row of the (J_perp, J_z) diagram."""
    K, jp, J_perp, Jz_over_K_list, Gamma, row_kwargs = args
    row_kwargs = dict(row_kwargs)
    verbose = row_kwargs.pop('verbose', True)
    points = [(K, J_perp, Gamma, jz * K) for jz in Jz_over_K_list]
    if verbose:
        print('[pid %d] starting row J_perp/|K|=%+.4f (%d Jz points)'
              % (os.getpid(), jp, len(Jz_over_K_list)), flush=True)
    # no per-point output from parallel workers
    row = continuation_scan_JperpJz(points, verbose=False, **row_kwargs)
    if verbose:
        print('[pid %d] finished row J_perp/|K|=%+.4f' % (os.getpid(), jp), flush=True)
    return jp, row


def build_phase_diagram_JperpJz_parallel(K, J_perp_over_K_list, Jz_over_K_list, Gamma_over_K=0.0,
                                         n_workers=None, checkpoint=True, checkpoint_path=None,
                                         resume=True, text_path=None, **kwargs):
    """Parallel (one process per row) version of build_phase_diagram_JperpJz, with checkpointing."""
    import multiprocessing as mp

    Gamma = Gamma_over_K * K
    verbose = kwargs.pop('verbose', True)
    Nx = kwargs.get('Nx', 18)
    Ny = kwargs.get('Ny', Nx)

    if checkpoint and checkpoint_path is None:
        checkpoint_path = _JperpJz_checkpoint_path(K, J_perp_over_K_list, Jz_over_K_list,
                                                    Gamma_over_K, Nx, Ny)

    results = {}
    if checkpoint and resume:
        loaded = _load_checkpoint(checkpoint_path, verbose=verbose)
        if loaded is not None:
            results = loaded
            if verbose:
                print('  [checkpoint] %d/%d row(s) already complete -- only solving the rest'
                      % (len(results), len(J_perp_over_K_list)))

    def _checkpoint_now():
        if checkpoint:
            _save_checkpoint(checkpoint_path, results)

    tasks = [(K, jp, jp * K, Jz_over_K_list, Gamma,
              dict(kwargs, verbose=verbose, text_path=text_path))
             for jp in J_perp_over_K_list if jp not in results]

    if not tasks:
        if verbose:
            print('build_phase_diagram_JperpJz_parallel: all %d row(s) already complete '
                  '(from checkpoint) -- nothing left to solve.' % len(J_perp_over_K_list))
        return {jp: results[jp] for jp in J_perp_over_K_list}

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(tasks)))

    if verbose:
        print('build_phase_diagram_JperpJz_parallel: %d row(s) remaining across %d worker '
              'process(es)' % (len(tasks), n_workers))

    if n_workers == 1:
        # No point spinning up a Pool for a single worker -- solve serially in-process
        # instead (also avoids fork/spawn startup overhead).
        for task in tasks:
            jp, row = _solve_JperpJz_row(task)
            results[jp] = {jz: pt for jz, pt in zip(Jz_over_K_list, row)}
            _checkpoint_now()
        return {jp: results[jp] for jp in J_perp_over_K_list}

    with mp.Pool(processes=n_workers) as pool:
        for jp, row in pool.imap_unordered(_solve_JperpJz_row, tasks):
            results[jp] = {jz: pt for jz, pt in zip(Jz_over_K_list, row)}
            _checkpoint_now()
            if verbose:
                print('  [main] %d/%d rows complete (just finished J_perp/|K|=%+.4f)'
                      % (len(results), len(J_perp_over_K_list), jp))

    # restore the original row order
    return {jp: results[jp] for jp in J_perp_over_K_list}


# Fixed colorblind-safe colour per phase label, used in every plot.
PATTERN_COLORS = {
    'QSL':                  '#2a78d6',  # blue   -- the quantum-disordered (m=0) reference state
    'QSL (bond-nematic!)':  '#eda100',  # yellow -- flagged/diagnostic variant of QSL, warning-coded
    'FM':                   '#4a3aa7',  # violet -- ordered, sublattices parallel
    'AFM':                 '#e34948',  # red    -- ordered, sublattices antiparallel
    'canted/other':         '#1baf7a',  # aqua   -- ordered but neither parallel nor antiparallel
}
_PATTERN_FALLBACK_COLOR = '#808080'  # gray -- defensive only; should never be hit in practice,
                                      # since the 5 keys above are the complete set classify_
                                      # solution() can return (see that method's own docstring)


def _pattern_color(pattern):
    """Fixed color for a classify_solution() pattern label; gray fallback for anything unknown
    (defensive -- classify_solution() only ever returns the 5 keys in PATTERN_COLORS)."""
    return PATTERN_COLORS.get(pattern, _PATTERN_FALLBACK_COLOR)


def _add_pattern_legend(ax, patterns_present, **legend_kwargs):
    """Legend with one colour patch per phase label present in the plot."""
    ordered = [p for p in PATTERN_COLORS if p in set(patterns_present)]
    handles = [mpatches.Patch(facecolor=_pattern_color(p), edgecolor='black', linewidth=0.5,
                               label=p) for p in ordered]
    kwargs = dict(fontsize=8, loc='best', frameon=True)
    kwargs.update(legend_kwargs)
    ax.legend(handles=handles, **kwargs)


def _pattern_grid_to_rgb(pattern_grid):
    """Convert a grid of phase labels to an RGB image using PATTERN_COLORS."""
    ny, nx = pattern_grid.shape
    rgb = np.zeros((ny, nx, 3))
    for i in range(ny):
        for j in range(nx):
            rgb[i, j, :] = matplotlib.colors.to_rgb(_pattern_color(pattern_grid[i, j]))
    return rgb


def _savefig_separate(fig, save, filename, default_path, suffix):
    """Save one figure as <base>_<suffix>_up.png with the CLEAN_MF_ prefix (save = 'y'/'n')."""
    if save == 'y':
        out = filename or default_path
        root, ext = os.path.splitext(out)
        out = _clean_mf_name('%s_%s%s%s' % (root, suffix, PLOT_NAME_TAG, ext or '.png'))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches='tight')
        return out
    elif save != 'n':
        raise NotImplementedError('wrong input. save = y,n')
    return None


def _is_ambiguous(winner, margin_tol=1e-4):
    """True if the winner is a bond-asymmetric QSL or has a near-degenerate runner-up (margin < margin_tol)."""
    pattern = winner.get('pattern', '')
    bond_sym_flag = bool(winner.get('bond_symmetry_broken', False)) and 'QSL' in pattern
    return bond_sym_flag or winner.get('energy_margin', float('inf')) < margin_tol


def _escalated_Nx(Nx, min_scale=1.5, multiple_of=6):
    """Next larger lattice size (multiple of 6, at least 1.5 x Nx) for re-solving ambiguous points."""
    target = max(Nx + multiple_of, int(np.ceil(Nx * min_scale)))
    return int(multiple_of * np.ceil(target / multiple_of))


def _flatten_points(scan_result):
    """Flatten any scan-result shape into a list of point dicts."""
    if isinstance(scan_result, dict):
        if 'winner' in scan_result:
            return [scan_result]
        out = []
        for v in scan_result.values():
            out.extend(_flatten_points(v))
        return out
    return list(scan_result)


def escalate_ambiguous_points(scan_result, margin_tol=1e-4, seed_kinds=None, m0=0.3,
                              base_seed=0, min_scale=1.5, max_rounds=2, verbose=True,
                              checkpoint_path=None, retry_unresolved=False):
    """Re-solve ambiguous points at larger Nx until they are unambiguous (in place, with checkpointing)."""
    if seed_kinds is None:
        seed_kinds = Majorana_rep.DEFAULT_SEED_KINDS

    points_iter = _flatten_points(scan_result)

    # split flagged points into already-escalated and to-do
    flagged = [pt for pt in points_iter if _is_ambiguous(pt['winner'], margin_tol=margin_tol)]
    to_process = []
    n_skipped_already_done = 0
    for pt in flagged:
        prior = pt.get('escalation')
        if prior is not None and (prior.get('resolved') or not retry_unresolved):
            n_skipped_already_done += 1
        else:
            to_process.append(pt)

    if verbose:
        print('[%s] escalate_ambiguous_points: %d ambiguous point(s) found total, %d already '
              'attempted (skipping), %d to process now'
              % (time.strftime('%H:%M:%S'), len(flagged), n_skipped_already_done,
                 len(to_process)))

    n_escalated = len(to_process)
    n_resolved = 0
    for i, pt in enumerate(to_process, 1):
        # points carry either 'delta' (Jz = J_perp*delta) or 'Jz'
        is_delta_point = 'delta' in pt
        if is_delta_point:
            Jz_value = pt['J_perp'] * pt['delta']
            axis_label, axis_value = 'Delta', pt['delta']
        else:
            Jz_value = pt['Jz']
            axis_label, axis_value = 'Jz', pt['Jz']

        winner = pt['winner']
        old_Nx = pt['Nx']
        old_pattern = winner['pattern']
        cur_Nx = old_Nx
        best = winner
        resolved = False
        rounds_used = 0
        if verbose:
            print('[%s] [escalate] point %d/%d: starting K=%+.3f J_perp=%+.4f Gamma=%+.4f '
                  '%s=%+.4f (current Nx=%d, pattern=%-20s)'
                  % (time.strftime('%H:%M:%S'), i, n_escalated, pt['K'], pt['J_perp'],
                     pt['Gamma'], axis_label, axis_value, old_Nx, old_pattern))
        for round_i in range(max_rounds):
            new_Nx = _escalated_Nx(cur_Nx, min_scale=min_scale)
            Kxyz = np.array([pt['K'], pt['K'], pt['K']])
            Jxyz = np.array([pt['J_perp'], pt['J_perp'], Jz_value])
            parameters = Parameters(new_Nx, new_Nx, Kxyz, Jxyz, pt['Gamma'])
            model = Majorana_rep(parameters)
            # multi-seed solve at the larger Nx
            best, all_results = model.solve_best_branch(
                seed_kinds=seed_kinds, m0=m0, base_seed=base_seed, verbose=verbose)
            cur_Nx = new_Nx
            rounds_used = round_i + 1
            resolved = not _is_ambiguous(best, margin_tol=margin_tol)
            if verbose:
                print('  [%s] [escalate] point %d/%d K=%+.3f J_perp=%+.4f Gamma=%+.4f '
                      '%s=%+.4f  round %d: Nx -> %d:  %-20s  (margin %.2e)%s'
                      % (time.strftime('%H:%M:%S'), i, n_escalated, pt['K'], pt['J_perp'],
                         pt['Gamma'], axis_label, axis_value, rounds_used, new_Nx,
                         best['pattern'], best['energy_margin'],
                         '' if resolved else '  still ambiguous'))
            if resolved:
                break

        n_resolved += int(resolved)
        pt['escalation'] = {'from_Nx': old_Nx, 'to_Nx': cur_Nx, 'rounds': rounds_used,
                            'old_pattern': old_pattern, 'new_pattern': best['pattern'],
                            'resolved': resolved}
        pt['Nx'] = pt['Ny'] = cur_Nx
        pt['branches'] = all_results
        pt['winner'] = best
        if verbose and not resolved:
            print('  [escalate] *** K=%+.3f J_perp=%+.4f Gamma=%+.4f %s=%+.4f STILL '
                  'AMBIGUOUS after %d round(s), Nx %d -> %d -- report as unresolved. ***'
                  % (pt['K'], pt['J_perp'], pt['Gamma'], axis_label, axis_value,
                     rounds_used, old_Nx, cur_Nx))

        if checkpoint_path is not None:
            # checkpoint after every point
            _save_checkpoint(checkpoint_path, scan_result)

    if verbose:
        print('escalate_ambiguous_points: %d point(s) flagged, %d resolved, %d still ambiguous '
              'after escalation, %d already-attempted point(s) skipped (use '
              'retry_unresolved=True to reattempt unresolved ones).'
              % (n_escalated, n_resolved, n_escalated - n_resolved, n_skipped_already_done))

    return scan_result


def summarize_scan(scan_result, axis_key='delta', e_tol=1e-6):
    """Table of winner, energy, gap and transition type along a 1D scan."""
    rows = []
    prev = None
    for pt in scan_result:
        winner = pt['winner']
        flag = ''
        if prev is not None and (winner['pattern'] != prev['pattern']
                                 or winner['seed'] != prev['seed']):
            same_seed_now = next((r for r in pt['branches'] if r['seed'] == prev['seed']), None)
            degenerate = (same_seed_now is not None and
                         abs(same_seed_now['variational_energy'] - winner['variational_energy']) < e_tol)
            d_m = abs(winner['|m_A|'] - prev['|m_A|'])
            flag = ('first-order (branch switch)' if (degenerate or d_m > 1e-2)
                   else 'branch switch -- inspect closely')
        rows.append({axis_key: pt[axis_key], 'pattern': winner['pattern'],
                    '|m_A|': winner['|m_A|'], 'gap': winner['gap'],
                    'seed': winner['seed'], 'flag': flag})
        prev = winner
    return rows


def regression_tests(K=1.0, Nx=18, Ny=18, verbose=True):
    """Sanity checks: pure-Kitaev fixed point, Hamiltonian consistency, extreme-Ising limit."""
    results = {}

    # 1. pure Kitaev point is an exact fixed point (raises on failure)
    ref = Majorana_rep(Parameters(Nx, Ny, np.array([K, K, K]), np.zeros(3), 0.0))
    ref.Kitaev_exact_MF_parameters()
    results['pure_Kitaev_fixed_point'] = True
    if verbose:
        print('[PASS] pure Kitaev point is an exact self-consistent fixed point')

    # 2. vectorized and per-k Hamiltonian / MF conditions must agree at a generic point
    rng = np.random.default_rng(2024)
    params = Parameters(Nx, Ny, K * np.array([1.0, 0.9, 1.1]), np.array([-0.1, 0.05, 0.2]), 0.3)
    model_check = Majorana_rep(params)
    x = rng.uniform(-1, 1, N_MF_PARAMETERS)
    res_vec = np.array(model_check.MF_conditions_general(x))
    res_ref = np.array(model_check._MF_conditions_general_scalar_reference(x))
    model_check.MF_parameters = x
    E_batched = model_check.GS_Energy()
    E_loop = -0.5 * sum(np.sum(LA.eigh(model_check.get_Hamiltonian(kx, ky, x))[0][4:8])
                        for kx in range(Nx) for ky in range(Ny)).real / (Nx * Ny)
    diff = max(float(np.max(np.abs(res_vec - res_ref))), abs(E_batched - E_loop))
    results['hamiltonian_consistency_max_diff'] = diff
    passed = diff < 1e-10
    results['hamiltonian_consistency_passed'] = passed
    if verbose:
        print('[%s] vectorized vs per-k Hamiltonian / MF conditions: max diff = %.2e'
              % ('PASS' if passed else 'FAIL -- investigate before trusting scans', diff))

    # 3. extreme Ising limit: saturated AFM along z, no bond correlations, energy linear in Jz
    K_small = 0.02 * K
    Jz_1, Jz_2 = 20.0 * K_small, 40.0 * K_small  # ratio exactly 2, both >> K_small

    def _solve_extreme_Ising(Jz):
        params = Parameters(Nx, Ny, np.array([K_small, K_small, K_small]),
                            np.array([0.0, 0.0, Jz]), 0.0)
        model = Majorana_rep(params)
        best, _ = model.solve_best_branch(m0=0.4, base_seed=1, verbose=False)
        return best

    win1 = _solve_extreme_Ising(Jz_1)
    win2 = _solve_extreme_Ising(Jz_2)

    sign_ok = (win1['pattern'] == 'AFM') and (win1['m_A'][2] * win1['m_B'][2] < 0)
    sat_ok = win1['|m_A|'] > 0.97 and win1['|m_B|'] > 0.97
    chi_max = max(abs(v) for v in win1['chi00_xyz'])
    decoupled_ok = chi_max < 0.05
    ratio = win2['variational_energy'] / win1['variational_energy'] if win1['variational_energy'] != 0 else np.nan
    linear_ok = abs(ratio - 2.0) < 0.05

    ising_passed = sign_ok and sat_ok and decoupled_ok and linear_ok
    results['extreme_Ising_limit'] = {
        'sign_ok': sign_ok, 'saturation_ok': sat_ok, 'decoupling_ok': decoupled_ok,
        'linearity_ok': linear_ok, 'passed': ising_passed,
        '|m_A|': win1['|m_A|'], 'chi_max': chi_max, 'E(2Jz)/E(Jz)': ratio,
    }
    if verbose:
        print('[%s] extreme Ising limit (K=%.4f, Jz=%.3f vs %.3f): pattern=%s '
              'mA_z*mB_z=%+.4f |m_A|=%.4f chi_max=%.2e E-ratio=%.4f (want ~2.0)'
              % ('PASS' if ising_passed else 'FAIL -- investigate before trusting scans',
                 K_small, Jz_1, Jz_2, win1['pattern'], win1['m_A'][2] * win1['m_B'][2],
                 win1['|m_A|'], chi_max, ratio))
        if not ising_passed:
            print('         sign_ok=%s saturation_ok=%s decoupling_ok=%s linearity_ok=%s'
                  % (sign_ok, sat_ok, decoupled_ok, linear_ok))

    return results


def plot_XXZ_scan(scan_result, save='n', filename=None, figsize=(6, 4)):
    """Plot |m_A| and the Majorana gap of the winning branch along a 1D scan."""
    deltas = [pt['delta'] for pt in scan_result]
    m_A = [pt['winner']['|m_A|'] for pt in scan_result]
    gaps = [pt['winner']['gap'] for pt in scan_result]
    patterns = [pt['winner']['pattern'] for pt in scan_result]
    ambiguous = [_is_ambiguous(pt['winner']) for pt in scan_result]
    amb_idx = [i for i, a in enumerate(ambiguous) if a]
    unique_patterns = sorted(set(patterns))
    default_path = os.getcwd() + '/paper_plots/XXZ_scan.png'
    saved_paths = []

    def _shade(ax):
        for i in range(len(deltas) - 1):
            ax.axvspan(deltas[i], deltas[i + 1], color=_pattern_color(patterns[i]), alpha=0.2)
        _add_pattern_legend(ax, unique_patterns)

    # --- panel 1: |m_A| ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(deltas, m_A, 'o-', color='black', zorder=4)
    ax1.set_ylabel(r'$|m_A|$ (winning branch)')
    ax1.set_xlabel(r'$\Delta = J_z / J_\perp$')
    _shade(ax1)
    # mark ambiguous points with open red circles
    if amb_idx:
        ax1.plot([deltas[i] for i in amb_idx], [m_A[i] for i in amb_idx],
                 'o', mfc='none', mec='red', markersize=11, mew=1.5, linestyle='none', zorder=5)
    fig1.tight_layout()
    saved_paths.append(_savefig_separate(fig1, save, filename, default_path, 'mA'))
    plt.show()

    # --- panel 2: gap ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(deltas, gaps, 'o-', color='darkred', zorder=4)
    ax2.set_ylabel(r'Majorana gap $/|K|$')
    ax2.set_xlabel(r'$\Delta = J_z / J_\perp$')
    _shade(ax2)
    if amb_idx:
        ax2.plot([deltas[i] for i in amb_idx], [gaps[i] for i in amb_idx],
                 'o', mfc='none', mec='red', markersize=11, mew=1.5, linestyle='none', zorder=5)
    fig2.tight_layout()
    saved_paths.append(_savefig_separate(fig2, save, filename, default_path, 'gap'))
    plt.show()

    return saved_paths


def plot_phase_diagram_2D(results_2d, delta, save='n', filename=None, figsize=(5.5, 4.5)):
    """Plot one Delta panel of build_phase_diagram_2D: phase, |m_A|, gap, m_A.m_B."""
    row_data = results_2d[delta]
    gammas = sorted(row_data.keys())
    j_perps = sorted(next(iter(row_data.values())).keys())

    pattern_grid = np.empty((len(gammas), len(j_perps)), dtype=object)
    gap_grid = np.zeros((len(gammas), len(j_perps)))
    m_grid = np.zeros((len(gammas), len(j_perps)))
    mAmB_grid = np.zeros((len(gammas), len(j_perps)))
    for i, g in enumerate(gammas):
        for j, jp in enumerate(j_perps):
            w = row_data[g][jp]['winner']
            pattern_grid[i, j] = w['pattern']
            gap_grid[i, j] = w['gap']
            m_grid[i, j] = w['|m_A|']
            mAmB_grid[i, j] = float(np.dot(w['m_A'], w['m_B']))

    unique_patterns = sorted(set(pattern_grid.flatten().tolist()), key=list(PATTERN_COLORS).index)
    default_path = os.getcwd() + '/paper_plots/phase_diagram_Delta%.3f.png' % delta
    saved_paths = []

    # --- panel 1: winning phase (fixed categorical colors, patch legend, no colorbar) ---
    fig0, ax0 = plt.subplots(figsize=figsize)
    rgb_grid = _pattern_grid_to_rgb(pattern_grid)
    ax0.pcolormesh(j_perps, gammas, rgb_grid, shading='nearest')
    ax0.set_title('winning phase (Delta=%.3f)' % delta)
    ax0.set_xlabel(r'$J_\perp / |K|$')
    ax0.set_ylabel(r'$\Gamma / |K|$')
    _add_pattern_legend(ax0, unique_patterns, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig0.tight_layout()
    saved_paths.append(_savefig_separate(fig0, save, filename, default_path, 'phase'))
    plt.show()

    # --- panel 2: |m_A| ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    im1 = ax1.pcolormesh(j_perps, gammas, m_grid, cmap='viridis', shading='nearest')
    ax1.set_title(r'$|m_A|$ (Delta=%.3f)' % delta)
    ax1.set_xlabel(r'$J_\perp / |K|$')
    ax1.set_ylabel(r'$\Gamma / |K|$')
    fig1.colorbar(im1, ax=ax1)
    fig1.tight_layout()
    saved_paths.append(_savefig_separate(fig1, save, filename, default_path, 'mA'))
    plt.show()

    # --- panel 3: Majorana gap ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    im2 = ax2.pcolormesh(j_perps, gammas, gap_grid, cmap='magma', shading='nearest')
    ax2.set_title('Majorana gap / |K| (Delta=%.3f)' % delta)
    ax2.set_xlabel(r'$J_\perp / |K|$')
    ax2.set_ylabel(r'$\Gamma / |K|$')
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    saved_paths.append(_savefig_separate(fig2, save, filename, default_path, 'gap'))
    plt.show()

    # panel 4: m_A . m_B (> 0 FM, < 0 AFM, 0 QSL)
    fig3, ax3 = plt.subplots(figsize=figsize)
    vmax = max(np.max(np.abs(mAmB_grid)), 1e-12)
    im3 = ax3.pcolormesh(j_perps, gammas, mAmB_grid, cmap='RdBu_r', shading='nearest',
                         vmin=-vmax, vmax=vmax)
    ax3.set_title(r'$\mathbf{m}_A\cdot\mathbf{m}_B$ (Delta=%.3f)' % delta)
    ax3.set_xlabel(r'$J_\perp / |K|$')
    ax3.set_ylabel(r'$\Gamma / |K|$')
    fig3.colorbar(im3, ax=ax3)
    fig3.tight_layout()
    saved_paths.append(_savefig_separate(fig3, save, filename, default_path, 'mAmB'))
    plt.show()

    return saved_paths


def plot_phase_diagram_JperpDelta(results_jd, Gamma_over_K=0.0, save='n', filename=None,
                                  figsize=(5.5, 4.5)):
    """Plot the (J_perp, Delta) diagram: phase, |m_A|, gap, m_A.m_B."""
    j_perps = sorted(results_jd.keys())
    deltas = sorted(next(iter(results_jd.values())).keys())

    pattern_grid = np.empty((len(j_perps), len(deltas)), dtype=object)
    gap_grid = np.zeros((len(j_perps), len(deltas)))
    m_grid = np.zeros((len(j_perps), len(deltas)))
    mAmB_grid = np.zeros((len(j_perps), len(deltas)))
    for i, jp in enumerate(j_perps):
        for j, d in enumerate(deltas):
            w = results_jd[jp][d]['winner']
            pattern_grid[i, j] = w['pattern']
            gap_grid[i, j] = w['gap']
            m_grid[i, j] = w['|m_A|']
            mAmB_grid[i, j] = float(np.dot(w['m_A'], w['m_B']))

    unique_patterns = sorted(set(pattern_grid.flatten().tolist()), key=list(PATTERN_COLORS).index)
    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpDelta_Gamma%.3f.png' % Gamma_over_K)
    saved_paths = []

    # --- panel 1: winning phase (fixed categorical colors, patch legend, no colorbar) ---
    fig0, ax0 = plt.subplots(figsize=figsize)
    rgb_grid = _pattern_grid_to_rgb(pattern_grid)
    ax0.pcolormesh(deltas, j_perps, rgb_grid, shading='nearest')
    ax0.set_title('winning phase (Gamma=%.3f|K|)' % Gamma_over_K)
    ax0.set_xlabel(r'$\Delta = J_z / J_\perp$')
    ax0.set_ylabel(r'$J_\perp / |K|$')
    _add_pattern_legend(ax0, unique_patterns, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig0.tight_layout()
    saved_paths.append(_savefig_separate(fig0, save, filename, default_path, 'phase'))
    plt.show()

    # --- panel 2: |m_A| ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    im1 = ax1.pcolormesh(deltas, j_perps, m_grid, cmap='viridis', shading='nearest')
    ax1.set_title(r'$|m_A|$ (Gamma=%.3f|K|)' % Gamma_over_K)
    ax1.set_xlabel(r'$\Delta = J_z / J_\perp$')
    ax1.set_ylabel(r'$J_\perp / |K|$')
    fig1.colorbar(im1, ax=ax1)
    fig1.tight_layout()
    saved_paths.append(_savefig_separate(fig1, save, filename, default_path, 'mA'))
    plt.show()

    # --- panel 3: Majorana gap ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    im2 = ax2.pcolormesh(deltas, j_perps, gap_grid, cmap='magma', shading='nearest')
    ax2.set_title('Majorana gap / |K| (Gamma=%.3f|K|)' % Gamma_over_K)
    ax2.set_xlabel(r'$\Delta = J_z / J_\perp$')
    ax2.set_ylabel(r'$J_\perp / |K|$')
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    saved_paths.append(_savefig_separate(fig2, save, filename, default_path, 'gap'))
    plt.show()

    # panel 4: m_A . m_B (> 0 FM, < 0 AFM, 0 QSL)
    fig3, ax3 = plt.subplots(figsize=figsize)
    vmax = max(np.max(np.abs(mAmB_grid)), 1e-12)
    im3 = ax3.pcolormesh(deltas, j_perps, mAmB_grid, cmap='RdBu_r', shading='nearest',
                         vmin=-vmax, vmax=vmax)
    ax3.set_title(r'$\mathbf{m}_A\cdot\mathbf{m}_B$ (Gamma=%.3f|K|)' % Gamma_over_K)
    ax3.set_xlabel(r'$\Delta = J_z / J_\perp$')
    ax3.set_ylabel(r'$J_\perp / |K|$')
    fig3.colorbar(im3, ax=ax3)
    fig3.tight_layout()
    saved_paths.append(_savefig_separate(fig3, save, filename, default_path, 'mAmB'))
    plt.show()

    return saved_paths


def plot_phase_diagram_JperpDelta_phase(results_jd, Gamma_over_K=0.0, save='n', filename=None,
                                        figsize=(5.5, 4.5)):
    """Winning-phase map of the (J_perp, Delta) diagram."""
    j_perps = sorted(results_jd.keys())
    deltas = sorted(next(iter(results_jd.values())).keys())

    pattern_grid = np.empty((len(j_perps), len(deltas)), dtype=object)
    for i, jp in enumerate(j_perps):
        for j, d in enumerate(deltas):
            w = results_jd[jp][d]['winner']
            pattern_grid[i, j] = w['pattern']

    unique_patterns = sorted(set(pattern_grid.flatten().tolist()), key=list(PATTERN_COLORS).index)
    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpDelta_Gamma%.3f.png' % Gamma_over_K)

    fig, ax = plt.subplots(figsize=figsize)
    rgb_grid = _pattern_grid_to_rgb(pattern_grid)
    ax.pcolormesh(deltas, j_perps, rgb_grid, shading='nearest')
    ax.set_title('winning phase (Gamma=%.3f|K|)' % Gamma_over_K)
    ax.set_xlabel(r'$\Delta = J_z / J_\perp$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    _add_pattern_legend(ax, unique_patterns, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'phase')
    plt.show()

    return saved_path


def plot_phase_diagram_JperpDelta_mAmB(results_jd, Gamma_over_K=0.0, save='n', filename=None,
                                       figsize=(5.5, 4.5)):
    """m_A . m_B map of the (J_perp, Delta) diagram."""
    j_perps = sorted(results_jd.keys())
    deltas = sorted(next(iter(results_jd.values())).keys())

    mAmB_grid = np.zeros((len(j_perps), len(deltas)))
    for i, jp in enumerate(j_perps):
        for j, d in enumerate(deltas):
            w = results_jd[jp][d]['winner']
            mAmB_grid[i, j] = float(np.dot(w['m_A'], w['m_B']))

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpDelta_Gamma%.3f.png' % Gamma_over_K)

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(np.max(np.abs(mAmB_grid)), 1e-12)
    im = ax.pcolormesh(deltas, j_perps, mAmB_grid, cmap='RdBu_r', shading='nearest',
                       vmin=-vmax, vmax=vmax)
    ax.set_title(r'$\mathbf{m}_A\cdot\mathbf{m}_B$ (Gamma=%.3f|K|)' % Gamma_over_K)
    ax.set_xlabel(r'$\Delta = J_z / J_\perp$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'mAmB')
    plt.show()

    return saved_path


def plot_phase_diagram_JperpDelta_magnetization_magnitudes(results_jd, Gamma_over_K=0.0,
                                                            save='n', filename=None,
                                                            figsize=(5.5, 4.5),
                                                            shared_scale=True):
    """|m_A| and |m_B| maps of the (J_perp, Delta) diagram."""
    j_perps = sorted(results_jd.keys())
    deltas = sorted(next(iter(results_jd.values())).keys())

    mA_grid = np.zeros((len(j_perps), len(deltas)))
    mB_grid = np.zeros((len(j_perps), len(deltas)))
    for i, jp in enumerate(j_perps):
        for j, d in enumerate(deltas):
            w = results_jd[jp][d]['winner']
            mA_grid[i, j] = w['|m_A|']
            mB_grid[i, j] = w['|m_B|']

    grids = {'mA_mag': mA_grid, 'mB_mag': mB_grid}
    labels = {'mA_mag': r'$|m_A|$', 'mB_mag': r'$|m_B|$'}

    if shared_scale:
        vmax_shared = max(max(np.max(g) for g in grids.values()), 1e-12)

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpDelta_Gamma%.3f.png' % Gamma_over_K)

    saved_paths = {}
    for key, grid in grids.items():
        vmax = vmax_shared if shared_scale else max(np.max(grid), 1e-12)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(deltas, j_perps, grid, cmap='viridis', shading='nearest',
                           vmin=0.0, vmax=vmax)
        ax.set_title('%s (Gamma=%.3f|K|)' % (labels[key], Gamma_over_K))
        ax.set_xlabel(r'$\Delta = J_z / J_\perp$')
        ax.set_ylabel(r'$J_\perp / |K|$')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        saved_paths[key] = _savefig_separate(fig, save, filename, default_path, key)
        plt.show()

    return saved_paths


def plot_phase_diagram_JperpDelta_magnetization_components(results_jd, Gamma_over_K=0.0,
                                                            save='n', filename=None,
                                                            figsize=(5.5, 4.5),
                                                            shared_scale=True):
    """Maps of the six components m^x,y,z_A,B of the (J_perp, Delta) diagram."""
    j_perps = sorted(results_jd.keys())
    deltas = sorted(next(iter(results_jd.values())).keys())

    # (sublattice, component index, axis label)
    components = [('A', 0, 'x'), ('A', 1, 'y'), ('A', 2, 'z'),
                  ('B', 0, 'x'), ('B', 1, 'y'), ('B', 2, 'z')]

    grids = {}
    for sub, idx, axis in components:
        key = 'm%s_%s' % (sub, axis)
        grid = np.zeros((len(j_perps), len(deltas)))
        for i, jp in enumerate(j_perps):
            for j, d in enumerate(deltas):
                w = results_jd[jp][d]['winner']
                m_vec = w['m_A'] if sub == 'A' else w['m_B']
                grid[i, j] = float(np.asarray(m_vec)[idx])
        grids[key] = grid

    if shared_scale:
        vmax_shared = max(max(np.max(np.abs(g)) for g in grids.values()), 1e-12)

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpDelta_Gamma%.3f.png' % Gamma_over_K)

    saved_paths = {}
    for sub, idx, axis in components:
        key = 'm%s_%s' % (sub, axis)
        grid = grids[key]
        vmax = vmax_shared if shared_scale else max(np.max(np.abs(grid)), 1e-12)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(deltas, j_perps, grid, cmap='RdBu_r', shading='nearest',
                           vmin=-vmax, vmax=vmax)
        ax.set_title(r'$m^{%s}_{%s}$ (Gamma=%.3f|K|)' % (axis, sub, Gamma_over_K))
        ax.set_xlabel(r'$\Delta = J_z / J_\perp$')
        ax.set_ylabel(r'$J_\perp / |K|$')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        saved_paths[key] = _savefig_separate(fig, save, filename, default_path, key)
        plt.show()

    return saved_paths


def plot_phase_diagram_JperpJz_phase(results_jj, Gamma_over_K=0.0, save='n', filename=None,
                                     figsize=(5.5, 4.5)):
    """Winning-phase map of the (J_perp, J_z) diagram."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    pattern_grid = np.empty((len(j_perps), len(jzs)), dtype=object)
    for i, jp in enumerate(j_perps):
        for j, jz in enumerate(jzs):
            w = results_jj[jp][jz]['winner']
            pattern_grid[i, j] = w['pattern']

    unique_patterns = sorted(set(pattern_grid.flatten().tolist()), key=list(PATTERN_COLORS).index)
    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    fig, ax = plt.subplots(figsize=figsize)
    rgb_grid = _pattern_grid_to_rgb(pattern_grid)
    ax.pcolormesh(jzs, j_perps, rgb_grid, shading='nearest')
    ax.set_title('winning phase (Gamma=%.3f|K|)' % Gamma_over_K)
    ax.set_xlabel(r'$J_z / |K|$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    _add_pattern_legend(ax, unique_patterns, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'phase')
    plt.show()

    return saved_path


def plot_phase_diagram_JperpJz_energy(results_jj, Gamma_over_K=0.0, save='n', filename=None,
                                      figsize=(5.5, 4.5)):
    """Variational energy per unit cell on the (J_perp, J_z) grid."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    E_grid = np.zeros((len(j_perps), len(jzs)))
    for i, jp in enumerate(j_perps):
        for j, jz in enumerate(jzs):
            E_grid[i, j] = results_jj[jp][jz]['winner']['variational_energy']

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(jzs, j_perps, E_grid, cmap='viridis', shading=PLOT_SHADING)
    ax.set_title('ground-state energy per unit cell (Gamma=%.3f|K|)' % Gamma_over_K)
    ax.set_xlabel(r'$J_z / |K|$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    fig.colorbar(im, ax=ax, label='variational energy')
    fig.tight_layout()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'energy')
    plt.show()

    return saved_path


def plot_phase_diagram_JperpJz_total_energy(results_jj, Gamma_over_K=0.0, save='n', filename=None,
                                            figsize=(5.5, 4.5), mark_points=None):
    """Total energy of the cluster (per unit cell x Nx*Ny) on the (J_perp, J_z) grid; optional marked points."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    E_grid = np.zeros((len(j_perps), len(jzs)))
    sizes = set()
    for i, jp in enumerate(j_perps):
        for j, jz in enumerate(jzs):
            pt = results_jj[jp][jz]
            n_cells = pt['Nx'] * pt['Ny']
            sizes.add((pt['Nx'], pt['Ny']))
            E_grid[i, j] = pt['winner']['variational_energy'] * n_cells

    if len(sizes) == 1:
        nx, ny = sizes.pop()
        size_txt = 'Nx=Ny=%d, N=%d spins' % (nx, 2 * nx * ny) if nx == ny else \
                   'Nx=%d, Ny=%d, N=%d spins' % (nx, ny, 2 * nx * ny)
    else:
        size_txt = 'mixed cluster sizes!'

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(jzs, j_perps, E_grid, cmap='viridis', shading=PLOT_SHADING)
    ax.set_title('total ground-state energy (%s, Gamma=%.3f|K|)' % (size_txt, Gamma_over_K))
    ax.set_xlabel(r'$J_z / |K|$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    fig.colorbar(im, ax=ax, label='total variational energy')
    for mp in (mark_points or []):
        i = int(np.argmin(np.abs(np.array(j_perps) - mp['J_perp'])))
        j = int(np.argmin(np.abs(np.array(jzs) - mp['Jz'])))
        ax.plot(jzs[j], j_perps[i], marker='*', color='red', markersize=12,
                markeredgecolor='white', markeredgewidth=0.8, zorder=5)
        text = 'MF: %.4f' % E_grid[i, j]
        if mp.get('label'):
            text += '\n' + mp['label']
        ax.annotate(text, xy=(jzs[j], j_perps[i]), xytext=(8, 8), textcoords='offset points',
                    fontsize=7, color='black', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='red', alpha=0.9))
    fig.tight_layout()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'total_energy')
    plt.show()

    return saved_path


def plot_phase_diagram_JperpJz_phase_3d(results_jj, Gamma_over_K=0.0, save='n', filename=None,
                                        figsize=(7, 6), elev=28, azim=-60):
    """3D winning-phase map: each phase drawn as a terrace at its own (non-physical) height."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    pattern_grid = np.empty((len(j_perps), len(jzs)), dtype=object)
    for i, jp in enumerate(j_perps):
        for j, jz in enumerate(jzs):
            pattern_grid[i, j] = results_jj[jp][jz]['winner']['pattern']

    unique_patterns = sorted(set(pattern_grid.flatten().tolist()), key=list(PATTERN_COLORS).index)
    height_of = {p: i for i, p in enumerate(PATTERN_COLORS)}
    Z = np.vectorize(lambda p: height_of.get(p, len(PATTERN_COLORS)))(pattern_grid).astype(float)

    ny, nx = pattern_grid.shape
    rgba_grid = np.ones((ny, nx, 4))
    for i in range(ny):
        for j in range(nx):
            rgba_grid[i, j, :3] = matplotlib.colors.to_rgb(_pattern_color(pattern_grid[i, j]))

    JZ, JP = np.meshgrid(jzs, j_perps)

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(JZ, JP, Z, facecolors=rgba_grid, shade=False, rstride=1, cstride=1,
                     linewidth=0, antialiased=False)
    ax.set_title('winning phase, 3D (Gamma=%.3f|K|)' % Gamma_over_K)
    ax.set_xlabel(r'$J_z / |K|$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    ax.set_zticks([])
    ax.set_zlabel('phase (height = arbitrary, not physical)')
    ax.view_init(elev=elev, azim=azim)
    _add_pattern_legend(ax, unique_patterns, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()
    # show before saving: the saved image keeps the view you rotated to
    plt.show()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'phase_3d')

    return saved_path


def plot_phase_diagram_JperpJz_total_energy_3d(results_jj, Gamma_over_K=0.0, save='n',
                                               filename=None, figsize=(7, 6), elev=28, azim=-60,
                                               mark_points=None):
    """3D surface of the total cluster energy over (J_z, J_perp); optional marked points."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    E_grid = np.zeros((len(j_perps), len(jzs)))
    sizes = set()
    for i, jp in enumerate(j_perps):
        for j, jz in enumerate(jzs):
            pt = results_jj[jp][jz]
            n_cells = pt['Nx'] * pt['Ny']
            sizes.add((pt['Nx'], pt['Ny']))
            E_grid[i, j] = pt['winner']['variational_energy'] * n_cells

    if len(sizes) == 1:
        nx, ny = sizes.pop()
        size_txt = 'Nx=Ny=%d, N=%d spins' % (nx, 2 * nx * ny) if nx == ny else \
                   'Nx=%d, Ny=%d, N=%d spins' % (nx, ny, 2 * nx * ny)
    else:
        size_txt = 'mixed cluster sizes!'

    JZ, JP = np.meshgrid(jzs, j_perps)

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(JZ, JP, E_grid, cmap='viridis', rstride=1, cstride=1,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.6, label='total variational energy')
    ax.set_title('total ground-state energy, 3D (%s, Gamma=%.3f|K|)' % (size_txt, Gamma_over_K))
    ax.set_xlabel(r'$J_z / |K|$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    ax.set_zlabel('total variational energy')
    ax.view_init(elev=elev, azim=azim)
    # lift marker and label above the surface so they are not hidden by it
    z_range = np.nanmax(E_grid) - np.nanmin(E_grid)
    z_lift_marker = 0.06 * z_range
    z_lift_text = 0.16 * z_range
    for mp in (mark_points or []):
        i = int(np.argmin(np.abs(np.array(j_perps) - mp['J_perp'])))
        j = int(np.argmin(np.abs(np.array(jzs) - mp['Jz'])))
        x0, y0, z0 = jzs[j], j_perps[i], E_grid[i, j]
        z_marker = z0 + z_lift_marker
        z_text = z0 + z_lift_text
        ax.plot([x0, x0], [y0, y0], [z0, z_text], color='red', linestyle='--',
                linewidth=1.3, zorder=9)
        ax.scatter([x0], [y0], [z_marker], marker='*', color='red', s=400,
                  edgecolor='black', linewidth=1.0, zorder=10, depthshade=False)
        text = 'MF: %.4f' % z0
        if mp.get('label'):
            text += '\n' + mp['label']
        ax.text(x0, y0, z_text, text, fontsize=9, color='black', weight='bold', zorder=11,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='red', alpha=0.9))
    fig.tight_layout()
    # show() BEFORE save -- see plot_phase_diagram_JperpJz_phase_3d()'s comment above for why.
    plt.show()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'total_energy_3d')

    return saved_path


def plot_phase_diagram_JperpJz_mAmB(results_jj, Gamma_over_K=0.0, save='n', filename=None,
                                    figsize=(5.5, 4.5)):
    """m_A . m_B map of the (J_perp, J_z) diagram."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    mAmB_grid = np.zeros((len(j_perps), len(jzs)))
    for i, jp in enumerate(j_perps):
        for j, jz in enumerate(jzs):
            w = results_jj[jp][jz]['winner']
            mAmB_grid[i, j] = float(np.dot(w['m_A'], w['m_B']))

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(np.max(np.abs(mAmB_grid)), 1e-12)
    im = ax.pcolormesh(jzs, j_perps, mAmB_grid, cmap='RdBu_r', shading=PLOT_SHADING,
                       vmin=-vmax, vmax=vmax)
    ax.set_title(r'$\mathbf{m}_A\cdot\mathbf{m}_B$ (Gamma=%.3f|K|)' % Gamma_over_K)
    ax.set_xlabel(r'$J_z / |K|$')
    ax.set_ylabel(r'$J_\perp / |K|$')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    saved_path = _savefig_separate(fig, save, filename, default_path, 'mAmB')
    plt.show()

    return saved_path


def plot_phase_diagram_JperpJz_magnetization_magnitudes(results_jj, Gamma_over_K=0.0,
                                                         save='n', filename=None,
                                                         figsize=(5.5, 4.5),
                                                         shared_scale=True):
    """|m_A| and |m_B| maps of the (J_perp, J_z) diagram."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    mA_grid = np.zeros((len(j_perps), len(jzs)))
    mB_grid = np.zeros((len(j_perps), len(jzs)))
    for i, jp in enumerate(j_perps):
        for j, jz in enumerate(jzs):
            w = results_jj[jp][jz]['winner']
            mA_grid[i, j] = w['|m_A|']
            mB_grid[i, j] = w['|m_B|']

    grids = {'mA_mag': mA_grid, 'mB_mag': mB_grid}
    labels = {'mA_mag': r'$|m_A|$', 'mB_mag': r'$|m_B|$'}

    if shared_scale:
        vmax_shared = max(max(np.max(g) for g in grids.values()), 1e-12)

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    saved_paths = {}
    for key, grid in grids.items():
        vmax = vmax_shared if shared_scale else max(np.max(grid), 1e-12)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(jzs, j_perps, grid, cmap='viridis', shading=PLOT_SHADING,
                           vmin=0.0, vmax=vmax)
        ax.set_title('%s (Gamma=%.3f|K|)' % (labels[key], Gamma_over_K))
        ax.set_xlabel(r'$J_z / |K|$')
        ax.set_ylabel(r'$J_\perp / |K|$')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        saved_paths[key] = _savefig_separate(fig, save, filename, default_path, key)
        plt.show()

    return saved_paths


def plot_phase_diagram_JperpJz_magnetization_components(results_jj, Gamma_over_K=0.0,
                                                         save='n', filename=None,
                                                         figsize=(5.5, 4.5),
                                                         shared_scale=True):
    """Maps of the six components m^x,y,z_A,B of the (J_perp, J_z) diagram."""
    j_perps = sorted(results_jj.keys())
    jzs = sorted(next(iter(results_jj.values())).keys())

    components = [('A', 0, 'x'), ('A', 1, 'y'), ('A', 2, 'z'),
                  ('B', 0, 'x'), ('B', 1, 'y'), ('B', 2, 'z')]

    grids = {}
    for sub, idx, axis in components:
        key = 'm%s_%s' % (sub, axis)
        grid = np.zeros((len(j_perps), len(jzs)))
        for i, jp in enumerate(j_perps):
            for j, jz in enumerate(jzs):
                w = results_jj[jp][jz]['winner']
                m_vec = w['m_A'] if sub == 'A' else w['m_B']
                grid[i, j] = float(np.asarray(m_vec)[idx])
        grids[key] = grid

    if shared_scale:
        vmax_shared = max(max(np.max(np.abs(g)) for g in grids.values()), 1e-12)

    default_path = (os.getcwd()
                     + '/paper_plots/phase_diagram_JperpJz_Gamma%.3f.png' % Gamma_over_K)

    saved_paths = {}
    for sub, idx, axis in components:
        key = 'm%s_%s' % (sub, axis)
        grid = grids[key]
        vmax = vmax_shared if shared_scale else max(np.max(np.abs(grid)), 1e-12)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(jzs, j_perps, grid, cmap='RdBu_r', shading=PLOT_SHADING,
                           vmin=-vmax, vmax=vmax)
        ax.set_title(r'$m^{%s}_{%s}$ (Gamma=%.3f|K|)' % (axis, sub, Gamma_over_K))
        ax.set_xlabel(r'$J_z / |K|$')
        ax.set_ylabel(r'$J_\perp / |K|$')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        saved_paths[key] = _savefig_separate(fig, save, filename, default_path, key)
        plt.show()

    return saved_paths


# Finite cluster: 4 x 4 unit cells x 2 sites = 32 spins.
Nx = Ny = 4
# Shading of the continuous maps: 'nearest' (one square per point) or 'gouraud' (smooth).
PLOT_SHADING = 'nearest'   # one sharp square per grid point; 'gouraud' = smooth interpolation
N_SPINS = 2 * Nx * Ny  # = 32
# Boundary-condition sector (0, 0.5): physical sector, gives the exact 32-spin Kitaev energy -6.282044.
KX_SHIFT, KY_SHIFT = 0.0, 0.5
system_parameters.kx_shift = KX_SHIFT
system_parameters.ky_shift = KY_SHIFT
Kxyz = np.array([1, 1, 1]) * (1)
Jxyz = np.array([1, 1, 1]) * (-0.09)  # isotropic Heisenberg reduces to the original scalar-J model
Gamma = 0

if __name__ == "__main__":
    # Step 0: regression tests (on the periodic grid)
    system_parameters.kx_shift, system_parameters.ky_shift = 0.0, 0.0
    regression_tests(K=1.0, Nx=Nx, Ny=Ny, verbose=True)  # run on the same 32-spin cluster
    system_parameters.kx_shift, system_parameters.ky_shift = KX_SHIFT, KY_SHIFT

    # Step 3: (J_perp, J_z) diagram at Gamma = 0 on the 32-spin cluster (parallel, checkpointed, text export).
    K_demo = 1.0
    J_PERP_RANGE_JZ = 0.25   # J_perp/|K| in [-0.25, 0.25]
    JZ_RANGE = 0.25          # J_z/|K| in [-0.25, 0.25]
    # grid: step 0.005 -> 101 x 101 points (~0.9 s per point per process)
    J_PERP_STEP_JZ = 0.005
    JZ_STEP = 0.005
    NX_NEAR_KITAEV = Nx      # = 4 -> 4 x 4 unit cells = 32 spins
    # escalation to larger Nx would leave the 32-spin cluster, so it is off
    ESCALATE = False
    J_perp_over_K_list_jz = np.round(
        np.arange(-J_PERP_RANGE_JZ, J_PERP_RANGE_JZ + J_PERP_STEP_JZ / 2, J_PERP_STEP_JZ), 6).tolist()
    Jz_over_K_list = np.round(
        np.arange(-JZ_RANGE, JZ_RANGE + JZ_STEP / 2, JZ_STEP), 6).tolist()
    Gamma_over_K_JJ = 0.0
    N_WORKERS = max(1, (os.cpu_count() or 4) - 2)
    CHECKPOINT_PATH_JJ = _JperpJz_checkpoint_path(
        K_demo, J_perp_over_K_list_jz, Jz_over_K_list, Gamma_over_K_JJ, Nx=NX_NEAR_KITAEV, Ny=NX_NEAR_KITAEV)
    TEXT_PATH_JJ = os.path.splitext(CHECKPOINT_PATH_JJ)[0] + '.txt'

    diagram_JperpJz = build_phase_diagram_JperpJz_parallel(
        K_demo, J_perp_over_K_list_jz, Jz_over_K_list, Gamma_over_K=Gamma_over_K_JJ,
        n_workers=N_WORKERS, Nx=NX_NEAR_KITAEV, Ny=NX_NEAR_KITAEV, verbose=True,
        checkpoint_path=CHECKPOINT_PATH_JJ, text_path=TEXT_PATH_JJ)

    if ESCALATE:
        print('\nEscalating ambiguous cells across the (J_perp/|K|, J_z/|K|) diagram at Gamma=0:')
        escalate_ambiguous_points(diagram_JperpJz, verbose=True,
                                  checkpoint_path=CHECKPOINT_PATH_JJ)
    else:
        n_amb = sum(1 for pt in _flatten_points(diagram_JperpJz) if _is_ambiguous(pt['winner']))
        print('\nEscalation disabled (fixed %d-spin cluster, Nx=Ny=%d): %d ambiguous point(s) left '
              'as is.' % (N_SPINS, NX_NEAR_KITAEV, n_amb))

    # final text table
    n_rows = export_JperpJz_results_to_text(diagram_JperpJz, TEXT_PATH_JJ)
    print('Wrote %d rows (ground-state energy + 27 MF parameters each) to %s'
          % (n_rows, TEXT_PATH_JJ))

    # Only the two 3D plots are made here; the 2D plots come from CLEAN_MF_N32_2d.py (same checkpoint).

    # --- 3D winning-phase map: each phase is a terrace at its own (arbitrary) height, so the
    # phase boundaries show up as visible steps.
    plot_phase_diagram_JperpJz_phase_3d(
        diagram_JperpJz, Gamma_over_K=Gamma_over_K_JJ, save='y',
        filename=f'{os.getcwd()}/paper_plots/phase_diagram_JperpJz_Gamma0.png')

    # 3D total energy of the cluster (x16); stars at J_perp = -0.125 (FM-Kitaev benchmark) and pure Kitaev
    MARK_POINT = [{'J_perp': -0.125, 'Jz': 0.0, 'label': ''},
                  {'J_perp': 0.0, 'Jz': 0.0, 'label': 'pure Kitaev'}]
    plot_phase_diagram_JperpJz_total_energy_3d(
        diagram_JperpJz, Gamma_over_K=Gamma_over_K_JJ, save='y',
        filename=f'{os.getcwd()}/paper_plots/phase_diagram_JperpJz_Gamma0.png',
        mark_points=MARK_POINT)
