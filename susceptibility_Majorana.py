import sys
import os
import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib as mpl

from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
import pickle

from database import Susceptibility_data_Majorana, Parameters

# Local data root: all intermediate/result files live under ./data next to the scripts,
# instead of the original hard-coded cluster path. Keep this identical across
# susceptibility_Majorana.py, susceptibility_parallel.py and susceptibility_path.py
# so writers and readers agree on where files live.
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


class system_parameters(object):

    def __init__(self, susceptibility_parameters, dtype=np.complex128):

        self.Nx = susceptibility_parameters.parameters.Nx
        self.Ny = susceptibility_parameters.parameters.Ny

        self.kx_basis = 2 * np.pi / self.Nx
        self.ky_basis = 2 * np.pi / self.Ny

        # Matrix for rotating to actual momenta
        self.bmatrix = np.array([[np.sqrt(3), -np.sqrt(3)], [1, 1], [0, 0]]) / 2
        # Matrix for rotating to lab frame in spin indices
        self.U_spin = np.array(
            [[1, 1, -2], [-np.sqrt(3), np.sqrt(3), 0], [np.sqrt(2), np.sqrt(2), np.sqrt(2)]]) / np.sqrt(6)

        self.dtype = dtype

    def momentum_function(self, kx, ky):

        expkx, expky, expkz = np.array([np.exp(1j * kx * self.kx_basis), np.exp(1j * ky * self.ky_basis), 1])

        return expkx, expky, expkz

    def path_BZ(self, susceptibility):
        """
        Take path in BZ Gamma->M->K-> Gamma

        :param array susceptibility: array dimension (Nx,Ny, omega (Nx))
        """
        susceptibility_path = np.zeros([self.Nx, self.Nx], dtype=self.dtype)

        for i in np.arange(self.Nx // 2):
            susceptibility_path[i] = susceptibility[i, i]
        # M -> K
        for i in np.arange(self.Nx // 6):
            susceptibility_path[self.Nx // 2 + i] = susceptibility[self.Nx // 2 - i - 1, self.Nx // 2 + i]
        # K -> Gamma
        for i in np.arange(self.Nx // 3):
            susceptibility_path[2 * self.Nx // 3 + i] = susceptibility[self.Nx // 3 - i, 2 * self.Nx // 3 - 1 - 2 * i]

        return susceptibility_path

    def path_BZ_1(self, susceptibility):
        """
        Take path in BZ Gamma->K->M1->K1->M->Gamma

        :param array susceptibility: array dimension (2Nx,Ny, omega (Nx)), (Nx,0) is Gamma
        """
        susceptibility_path = np.zeros([2 * self.Nx, self.Nx], dtype=self.dtype)
        # Gamma->K ->M1
        for i in np.arange(self.Nx // 2):
            susceptibility_path[i] = susceptibility[self.Nx - i, i]
        # M1 -> K1
        for i in np.arange(self.Nx // 2):
            susceptibility_path[self.Nx // 2 + i] = susceptibility[self.Nx // 2 + i, self.Nx // 2 + i]
        # K1 -> M
        for i in np.arange(self.Nx // 2):
            susceptibility_path[self.Nx + i] = susceptibility[self.Nx + i - 1, self.Nx - i - 1]
        # M->Gamma
        for i in np.arange(self.Nx // 2):
            susceptibility_path[3 * self.Nx // 2 + i] = susceptibility[3 * self.Nx // 2 - 1 - i, self.Nx // 2 - i - 1]

        return susceptibility_path

    def path_BZ_2(self, susceptibility):
        """
        Take path in BZ Gamma->M->K1->M_1->K_1'

        :param array susceptibility: array dimension (2Nx,Ny, omega (Nx)), (Nx,0) is Gamma
        """
        susceptibility_path = np.zeros([2 * self.Nx, self.Nx], dtype=self.dtype)
        # Gamma->M-> K_1 ->M1->K_1'
        for i in np.arange(self.Nx):
            susceptibility_path[i] = susceptibility[self.Nx + i, 0]
        # M1 -> K1
        for i in np.arange(self.Nx):
            susceptibility_path[self.Nx + i] = susceptibility[2 * self.Nx - 1, i]

        return susceptibility_path


class Susceptibility_evaluation(system_parameters):

    def __init__(self, susceptibility_parameters, data_type, U=1, dtype=np.complex128):
        super().__init__(susceptibility_parameters)

        self.omega_max = susceptibility_parameters.omega_max
        self.delta = susceptibility_parameters.delta
        self.len_chucks = susceptibility_parameters.len_chunks
        self.N_core = susceptibility_parameters.N_core

        self.Nx = susceptibility_parameters.parameters.Nx
        self.Ny = susceptibility_parameters.parameters.Ny
        self.omega_step = self.omega_max / self.Nx

        self.Kxyz = susceptibility_parameters.parameters.Kxyz
        self.Jxyz = susceptibility_parameters.parameters.Jxyz
        self.Gamma = susceptibility_parameters.parameters.Gamma
        self.epsilon = susceptibility_parameters.parameters.epsilon

        self.h_dir = susceptibility_parameters.h_dir
        self.h_field = susceptibility_parameters.h_field

        self.U = U
        self.data_type = data_type
        self.dtype = dtype

        self.load_type = {'free': 0, 'RPA': f'RPA({self.U}U)'}

    def pack_path(self, h_dir):

        if h_dir == 'c':
            load_path = f'{LOCAL_DATA_DIR}/h(111)/K{self.Kxyz[0]}J{list(self.Jxyz)}Gamma{self.Gamma}eps{list(self.epsilon)}/'
            save_path = f'{LOCAL_DATA_DIR}/results/h(111)/K{self.Kxyz[0]}J{list(self.Jxyz)}Gamma{self.Gamma}eps{list(self.epsilon)}/'

        elif h_dir == 'b':
            load_path = f'{LOCAL_DATA_DIR}/h(1bar10)/K{self.Kxyz[0]}J{list(self.Jxyz)}Gamma{self.Gamma}eps{list(self.epsilon)}/'
            save_path = f'{LOCAL_DATA_DIR}/results/h(1bar10)/K{self.Kxyz[0]}J{list(self.Jxyz)}Gamma{self.Gamma}eps{list(self.epsilon)}/'
        else:
            raise NotImplementedError('incorrect field directions')

        return load_path, save_path

    def pack_susceptibility_adiabatic(self):
        load_path, save_path = self.pack_path(self.h_dir)
        self.Interaction_vertices()

        susceptibility_RPA = np.zeros([2 * self.Nx, self.Nx], dtype=self.dtype)
        k_chunks = np.load(f'{LOCAL_DATA_DIR}/MomentumChunks_path({self.Nx},{self.Ny}).npy', allow_pickle=True)
        assert len(k_chunks) == 2 * self.Nx
        for i in np.arange(2 * self.Nx):
            susceptibility_array = np.load(
                f'{load_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_path{i}.npy'
            )

            kx, ky = k_chunks[i]

            susceptibility_spin_RPA = self.RPA_kernel(kx, ky, susceptibility_array)
            susceptibility_RPA[i] = self.susceptibility_spin_basis(susceptibility_spin_RPA, kx, ky)[:, 2, 2]

            np.save(
                f'{save_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_RPA_projected_path_Szz.npy',
                susceptibility_RPA)

    def pack_susceptibility_projected(self, pack_type='both'):
        pack_type_catalogue = {'free', 'RPA', 'both'}
        if pack_type not in pack_type_catalogue:
            raise NotImplementedError(
                f'incorrect susceptibility pack_type designation: must be one of \' {list(pack_type_catalogue)} \' ')

        load_path, save_path = self.pack_path(self.h_dir)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        self.Interaction_vertices()

        if self.data_type == 'path':
            susceptibility_0 = np.zeros([2 * self.Nx, self.Nx], dtype=self.dtype)
            susceptibility_RPA = np.zeros([2 * self.Nx, self.Nx], dtype=self.dtype)
            k_chunks = np.load(f'{LOCAL_DATA_DIR}/MomentumChunks_path({self.Nx},{self.Ny}).npy', allow_pickle=True)
            assert len(k_chunks) == 2 * self.Nx
            for i in np.arange(2 * self.Nx):
                susceptibility_array = np.load(
                    f'{load_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_path{i}.npy'
                )

                kx, ky = k_chunks[i]
                if pack_type == 'both' or pack_type == 'free':
                    susceptibility_0[i] = self.susceptibility_spin_project(susceptibility_array, kx, ky)

                if pack_type == 'both' or pack_type == 'RPA':
                    susceptibility_spin_RPA = self.RPA_kernel(kx, ky, susceptibility_array)
                    susceptibility_RPA[i] = self.susceptibility_spin_project(susceptibility_spin_RPA, kx, ky)

            load_label_1 = self.load_type['free']
            load_label_2 = self.load_type['RPA']

            if pack_type == 'both' or pack_type == 'free':
                np.save(
                    f'{save_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{load_label_1}_projected_path.npy',
                    susceptibility_0)
            if pack_type == 'both' or pack_type == 'RPA':
                np.save(
                    f'{save_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{load_label_2}_projected_path.npy',
                    susceptibility_RPA)

        else:
            susceptibility_0 = np.zeros([2 * self.Nx, 2 * self.Ny, self.Nx], dtype=self.dtype)
            susceptibility_RPA = np.zeros([2 * self.Nx, 2 * self.Ny, self.Nx], dtype=self.dtype)
            k_chunks = np.load(f'{LOCAL_DATA_DIR}/MomentumChunks({self.Nx},{self.Ny}).npy', allow_pickle=True)
            N_batch = self.Nx * self.Ny // self.len_chucks
            assert self.len_chucks == len(k_chunks[0])
            for i in np.arange(N_batch):
                if self.data_type == 'BZ':
                    susceptibility_array = np.load(
                        f'{load_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_0{i}.npy'
                    )
                elif self.data_type == 'frequency':
                    susceptibility_array = np.load(
                        f'{load_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_const{i}.npy'
                    )
                for j in np.arange(len_chunks):
                    kx, ky = k_chunks[i, j]
                    susceptibility_temp = susceptibility_array[j]
                    if pack_type == 'both' or pack_type == 'free':
                        susceptibility_spin_1 = self.susceptibility_spin_project(susceptibility_temp, -self.Nx + kx,
                                                                                 -self.Ny + ky)
                        susceptibility_spin_2 = self.susceptibility_spin_project(susceptibility_temp, -self.Nx + kx, ky)
                        susceptibility_spin_3 = self.susceptibility_spin_project(susceptibility_temp, kx, -self.Ny + ky)
                        susceptibility_spin_4 = self.susceptibility_spin_project(susceptibility_temp, kx, ky)

                        susceptibility_0[kx, ky] = susceptibility_spin_1
                        susceptibility_0[kx, ky + self.Ny] = susceptibility_spin_2
                        susceptibility_0[kx + self.Nx, ky] = susceptibility_spin_3
                        susceptibility_0[kx + self.Nx, ky + self.Ny] = susceptibility_spin_4

                    if pack_type == 'both' or pack_type == 'RPA':
                        susceptibility_RPA_temp = self.RPA_kernel(kx, ky, susceptibility_temp)
                        susceptibility_spin_RPA_1 = self.susceptibility_spin_project(susceptibility_RPA_temp,
                                                                                     -self.Nx + kx, -self.Ny + ky)
                        susceptibility_spin_RPA_2 = self.susceptibility_spin_project(susceptibility_RPA_temp,
                                                                                     -self.Nx + kx, ky)
                        susceptibility_spin_RPA_3 = self.susceptibility_spin_project(susceptibility_RPA_temp, kx,
                                                                                     -self.Ny + ky)
                        susceptibility_spin_RPA_4 = self.susceptibility_spin_project(susceptibility_RPA_temp, kx, ky)
                        susceptibility_RPA[kx, ky] = susceptibility_spin_RPA_1
                        susceptibility_RPA[kx, ky + self.Ny] = susceptibility_spin_RPA_2
                        susceptibility_RPA[kx + self.Nx, ky] = susceptibility_spin_RPA_3
                        susceptibility_RPA[kx + self.Nx, ky + self.Ny] = susceptibility_spin_RPA_4

            load_label_1 = self.load_type['free']
            load_label_2 = self.load_type['RPA']
            if self.data_type == 'BZ':
                if pack_type == 'both' or pack_type == 'free':
                    np.save(
                        f'{save_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{load_label_1}_projected_extended_BZ.npy',
                        susceptibility_0)
                if pack_type == 'both' or pack_type == 'RPA':
                    np.save(
                        f'{save_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{load_label_2}_projected_extended_BZ.npy',
                        susceptibility_RPA)
            elif self.data_type == 'frequency':
                if pack_type == 'both' or pack_type == 'free':
                    np.save(
                        f'{save_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{load_label_1}_projected_const.npy',
                        susceptibility_0[0])
                if pack_type == 'both' or pack_type == 'RPA':
                    np.save(
                        f'{save_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{load_label_2}_projected_const.npy',
                        susceptibility_RPA[0])

    def Interaction_vertices(self):
        """
        Real-space interaction vertices U_{i alpha, j beta}(bond) entering the RPA kernel,
        cf. kitaev-electric-13_07_2026.tex Eq. (H_gen)/(HammMaj3): index convention is
        U[bond w in {x,y,z}, sublattice i, spin a, sublattice j, spin b].

        Anisotropic Heisenberg: (K^gamma + J^gamma) on the diagonal gamma-component of the
        gamma-bond, J^beta on the two remaining diagonal components (beta != gamma).

        Electric field: the off-diagonal Gamma coupling is replaced by
            Lambda_gamma^{alpha beta} = Gamma + eps_gamma   ((alpha,beta,gamma) cyclic)
                                       = Gamma - eps_gamma   (otherwise),
        which is antisymmetric under alpha<->beta exchange for fixed gamma (matching the
        Gamma +/- eps_gamma entries of the J_<ij>_gamma matrices in the tex file).
        Setting epsilon = (0,0,0) recovers the pure KJGamma model.
        """
        Jx, Jy, Jz = self.Jxyz
        Gamma = self.Gamma
        eps_x, eps_y, eps_z = self.epsilon
        Lambda_x_yz, Lambda_x_zy = Gamma + eps_x, Gamma - eps_x
        Lambda_y_zx, Lambda_y_xz = Gamma + eps_y, Gamma - eps_y
        Lambda_z_xy, Lambda_z_yx = Gamma + eps_z, Gamma - eps_z

        # Kitaev coupling (K^gamma+J^gamma), 3: bond indices (x, y, z)
        U_K = np.zeros((3, 2, 3, 2, 3), dtype=self.dtype)
        U_K[0, 0, 0, 1, 0] = self.Kxyz[0] + Jx
        U_K[1, 0, 1, 1, 1] = self.Kxyz[1] + Jy
        U_K[2, 0, 2, 1, 2] = self.Kxyz[2] + Jz

        # Heisenberg coupling: J^beta on the two components beta != gamma of the gamma-bond
        U_J = np.zeros((3, 2, 3, 2, 3), dtype=self.dtype)
        # x-bond
        U_J[0, 0, 1, 1, 1] = Jy
        U_J[0, 0, 2, 1, 2] = Jz
        # y-bond
        U_J[1, 0, 0, 1, 0] = Jx
        U_J[1, 0, 2, 1, 2] = Jz
        # z-bond
        U_J[2, 0, 0, 1, 0] = Jx
        U_J[2, 0, 1, 1, 1] = Jy

        # Gamma + electric-field coupling: Lambda_gamma^{alpha beta}
        U_Gamma = np.zeros((3, 2, 3, 2, 3), dtype=self.dtype)
        # x-bond: (alpha,beta) = (y,z), (z,y)
        U_Gamma[0, 0, 1, 1, 2] = Lambda_x_yz
        U_Gamma[0, 0, 2, 1, 1] = Lambda_x_zy
        # y-bond: (alpha,beta) = (x,z), (z,x)
        U_Gamma[1, 0, 0, 1, 2] = Lambda_y_xz
        U_Gamma[1, 0, 2, 1, 0] = Lambda_y_zx
        # z-bond: (alpha,beta) = (x,y), (y,x)
        U_Gamma[2, 0, 0, 1, 1] = Lambda_z_xy
        U_Gamma[2, 0, 1, 1, 0] = Lambda_z_yx

        self.Interaction_array = U_K, U_J, U_Gamma

    def RPA_kernel(self, kx, ky, susceptibility):

        # U_f = Interaction_vertices
        U_K, U_J, U_Gamma = self.Interaction_array

        expkx, expky, expkz = self.momentum_function(kx, ky)
        U_total = np.einsum('w,wiajb->iajb', np.array([expkx, expky, expkz]), U_K + U_Gamma + U_J) * self.U / 4
        U_total += U_total.transpose(2, 3, 0, 1).conj()

        Kernel = np.einsum('iajb,wjbkc-> wiakc', U_total, susceptibility)

        Inverse_Kernel = np.array(
            [LA.inv(np.eye(6) + Kernel[omega].reshape(6, 6)) for omega in np.arange(self.Nx)]).reshape(self.Nx, 2, 3, 2,
                                                                                                       3)

        susceptibility_array_RPA = np.einsum('wiajb,wjbkc->wiakc', susceptibility, Inverse_Kernel)

        return susceptibility_array_RPA

    def susceptibility_spin_basis(self, susceptibility, kx, ky):
        # susceptibility in spin basis for given momentum component

        expkz = np.exp(-1j * (self.kx_basis * kx + self.ky_basis * ky) / 3)

        momentum_array = np.array([[1, expkz], [expkz.conj(), 1]])

        # array in spin indices only, contracting sublattice parts
        susceptibility_spin_lab = np.einsum('wiajb,ij->wab', susceptibility, momentum_array)

        susceptibility_spin = np.einsum('aj,wjk,kb->wab', self.U_spin, susceptibility_spin_lab, self.U_spin.T)

        return susceptibility_spin

    def susceptibility_spin_project(self, susceptibility, kx, ky):
        # susceptibility in spin basis for given momentum component

        susceptibility_spin = self.susceptibility_spin_basis(susceptibility, kx, ky)

        if kx == ky == 0:
            kvec = np.dot(self.bmatrix, np.array([kx + 1, ky + 1])) / self.Nx
        else:
            kvec = np.dot(self.bmatrix, np.array([kx, ky])) / self.Nx

        kvec_norm = np.linalg.norm(kvec)

        polarisation = np.eye(3) - np.tensordot(kvec, kvec, axes=0) / kvec_norm ** 2

        return np.einsum('ij,wij->w', polarisation, susceptibility_spin)

    def load_susceptibility(self, load_type):

        if load_type not in self.load_type.keys():
            raise NotImplementedError(
                f'incorrect susceptibility load_type designation: must be one of \' {list(self.load_type.keys())} \' ')

        self.load_label = self.load_type[load_type]
        load_path = self.pack_path(self.h_dir)[1]

        if self.data_type == 'BZ':
            self.susceptibility = np.load(
                f'{load_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{self.load_label}_projected_extended_BZ.npy')
        elif self.data_type == 'path':
            self.susceptibility = np.load(
                f'{load_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{self.load_label}_projected_path.npy')
        elif self.data_type == 'frequency':
            self.susceptibility = np.load(
                f'{load_path}h={self.h_field}(Nx{self.Nx}omegamax{self.omega_max}delta{self.delta})susceptibility_Majorana_{self.load_label}_projected_const.npy')

        self.save_type_name()

    def plot_susceptibility_path_1stBZ(self, save='n'):
        if self.data_type == 'BZ':
            # prepare array for paths in 1st BZ
            susceptibility_path = self.path_BZ(self.susceptibility[self.Nx:2 * self.Nx, self.Ny:2 * self.Ny])
        elif self.data_type == 'path':
            susceptibility_path = self.susceptibility[0:self.Nx]
        else:
            raise NotImplementedError('wrong data_type')

        max_value = np.delete(susceptibility_path.imag, [0], axis=1).max()
        min_value = susceptibility_path.imag.min()
        chi_pm = plt.contourf(np.arange(self.Nx), np.arange(self.Ny), susceptibility_path.imag.T,
                              levels=np.linspace(min_value, max_value, 300),
                              cmap='viridis', extend='both')
        cbar = plt.colorbar(chi_pm, ticks=np.linspace(min_value, max_value, 5))
        plt.xticks(np.array([0, self.Nx // 2, 2 * self.Nx // 3, self.Nx - 1]),
                   [r" $\Gamma$", r" M", r" K", r" $\Gamma$"])
        plt.yticks(np.array([0, self.Nx // 2, self.Nx - 1]), np.array([0, self.omega_max / 2, self.omega_max]))
        chi_pm.changed()
        plt.title(
            f'$\\mathcal{{I}}_{{{self.load_label}}}, (K,J,\\Gamma,\\epsilon) = {self.Kxyz[0], list(self.Jxyz), self.Gamma, list(self.epsilon)}, h = {self.h_field}{self.plot_title}$')

        if save == 'y':
            plt.savefig(f'{self.save_path}_projected.png', dpi=300, format='png')
        elif save == 'n':
            pass
        else:
            raise NotImplementedError('wrong input. save = y,n  ')

        plt.show()

    def plot_susceptibility_path_2ndBZ(self, save='n'):

        if self.data_type == 'BZ':
            # prepare array for paths in 2nd BZ, only half of ky values are taken
            susceptibility_path = self.path_BZ_1(self.susceptibility[0:2 * self.Nx, self.Ny:2 * self.Ny])
        elif self.data_type == 'path':
            susceptibility_path = self.susceptibility
        else:
            raise NotImplementedError('wrong data_type')

        max_value = np.delete(susceptibility_path.imag, [0], axis=1).max() / 2
        min_value = susceptibility_path.imag.min()
        chi_pm = plt.contourf(np.arange(2 * self.Nx), np.arange(self.Ny), susceptibility_path.imag.T,
                              levels=np.linspace(min_value, max_value, 300),
                              cmap='viridis', extend='both')
        cbar = plt.colorbar(chi_pm, ticks=np.linspace(min_value, max_value, 5))

        plt.xlim(np.array([0, 2 * self.Nx]))
        plt.ylim(np.array([0, self.Nx]))

        plt.xticks(
            np.array([0, self.Nx // 3, self.Nx // 2, self.Nx, 4 * self.Nx // 3, 3 * self.Nx // 2, 2 * self.Nx - 1]),
            [r" $\Gamma$", r"K", r" M$_1$", r"K$_1$", r" K'", r" M", r" $\Gamma$"])
        plt.yticks(np.array([0, self.Nx // 2, self.Nx - 1]), np.array([0, self.omega_max / 2, self.omega_max]))

        chi_pm.changed()
        plt.title(
            f'$\\mathcal{{I}}_{{{self.load_label}}}, (K,J,\\Gamma,\\epsilon) = {self.Kxyz[0], list(self.Jxyz), self.Gamma, list(self.epsilon)}, h = {self.h_field}{self.plot_title}$')
        if save == 'y':
            plt.savefig(f'{self.save_path}_projected.png', dpi=300, format='png')
        elif save == 'n':
            pass
        else:
            raise NotImplementedError('wrong input. save = y,n  ')

        plt.show()

    def plot_susceptibility_momentum_cut(self, kx=0, ky=0, save='n'):

        if self.data_type == 'BZ':
            if kx == self.Nx and ky == self.Ny:
                momentum_label = f'\\mathbf{{k}} = 0'
            elif kx == self.Nx + self.Nx // 3 and ky == self.Ny + 2 * self.Nx // 3:
                momentum_label = f'\\mathbf{{k}} = \\Gamma'
            else:
                k1, k2, k3 = np.einsum('ij,j->i', self.bmatrix, np.array([kx, ky]) / self.Nx)
                momentum_label = f'\\mathbf{{k}} = {k1, k2}'

            plt.plot(np.arange(self.Nx), self.susceptibility[kx, ky].imag.T)

        elif self.data_type == 'path':
            if kx == self.Nx and ky == self.Ny:
                momentum_label = f'\\mathbf{{k}} = 0'
                plt.plot(np.arange(self.Nx), self.susceptibility[0].imag.T)

            else:
                NotImplementedError(f'Only k=0 can be chosen right now.')
        else:
            raise NotImplementedError('wrong data_type')

        plt.xlabel(r"$\omega$")
        plt.ylabel(f"$\\mathcal{{I}}_{{{self.load_label}}}$")
        plt.xticks(np.array([0, self.Nx // 2, self.Nx - 1]), np.array([0, self.omega_max / 2, self.omega_max]))
        plt.xlim(0, self.Nx - 1)
        plt.ylim(bottom=0)
        plt.title(
            f'$\\mathcal{{I}}_{{{self.load_label}}}(\\omega,{momentum_label}),  (K,J,\\Gamma,\\epsilon) = {self.Kxyz[0], list(self.Jxyz), self.Gamma, list(self.epsilon)}, h = {self.h_field}{self.plot_title}$')
        if save == 'y':
            plt.savefig(f'{self.save_path}_projected(cut{kx},{ky}).png', dpi=300, format='png')
        elif save == 'n':
            pass
        else:
            raise NotImplementedError('wrong input. save = y,n  ')

        plt.show()

    def compute_free_susceptibility_k(self, Ek, uk, kx=0, ky=0, rpa=False):
        """
        Compute the BARE (pre-RPA) Majorana-bubble susceptibility tensor chi_0(omega, k) at a
        single momentum point (kx, ky), directly from a Majorana MFT eigensystem (Ek, uk) --
        e.g. from KJGamma_MF_E_aniso.Majorana_rep.get_eiegensystem_Majorana() -- with NO disk
        I/O (unlike pack_susceptibility_projected()/susceptibility_parallel.py, which are
        built for a full-BZ, multi-process, file-based pipeline that is unnecessary overhead
        for a single k-point cut). This is the direct MFT-level analogue of the paper's
        Eq. (24)/(47) -- i.e. the un-dressed response shown in Fig. 4(b)'s inset of Rao,
        Moessner & Knolle (PRB 112, 024440, 2025) -- evaluated at k=(kx,ky) across the full
        omega grid.

        Ek, uk : the (Nx,Ny,4) and (Nx,Ny,8,4) MFT eigensystem arrays for THIS model, i.e.
            Ek, uk = model.get_eiegensystem_Majorana() where `model` is a converged
            Majorana_rep instance (from KJGamma_MF_E_aniso.py) with the SAME Nx, Ny, Kxyz,
            Jxyz, Gamma, epsilon, h_dir, h_field as self. Works unchanged for genuinely
            anisotropic Jxyz / epsilon -- the underlying bare-bubble formalism (Eq. 24) never
            assumed isotropic J; only the choice of which diagonal spin components to plot
            (see plot_free_susceptibility_momentum_cut) needs adjusting once Jx != Jy != Jz,
            since they're no longer forced equal by symmetry.
        rpa : bool, default False
            If True, ALSO apply the RPA dressing (Eq. 25, via RPA_kernel -- the same
            interaction vertices used by pack_susceptibility_projected(pack_type='RPA'), just
            applied here directly to a single in-memory k-point instead of a saved/packed
            full-BZ array) to the SAME raw bubble computed above (no redundant recomputation),
            and return (chi_0, chi_RPA) instead of just chi_0.

        Returns the (Nx_omega, 3, 3) array chi_0^{ab}(omega, kx, ky) (or the (chi_0, chi_RPA)
        pair if rpa=True) in the same (a,b,c) lab-frame spin basis as susceptibility_spin_basis()
        / the rest of this file's RPA pipeline (so it is directly comparable to a packed cut of
        the same component).
        """
        free_parameters = Susceptibility_parameters(
            Parameters(self.Nx, self.Ny, self.Kxyz, self.Jxyz, self.Gamma, self.epsilon),
            self.h_field, self.h_dir, self.len_chucks, self.omega_max, self.delta,
            self.N_core, Ek, uk)
        tensor = Compute_susceptibility_tensor(free_parameters, omega=None, dtype=self.dtype)
        susceptibility_0_raw = tensor.calculate_susceptibility_free_Parallel(kx, ky)[0]  # (Nx,2,3,2,3)

        chi_0 = self.susceptibility_spin_basis(susceptibility_0_raw, kx, ky)
        if not rpa:
            return chi_0

        self.Interaction_vertices()
        susceptibility_RPA_raw = self.RPA_kernel(kx, ky, susceptibility_0_raw)
        chi_RPA = self.susceptibility_spin_basis(susceptibility_RPA_raw, kx, ky)
        return chi_0, chi_RPA

    def compute_INS_intensity_k(self, Ek, uk, kx, ky, rpa=True):
        """
        Compute the polarization-projected INS intensity I(omega, k) = [delta_ab -
        k_a k_b/k^2] Im chi^{ab}(omega, k) -- Eq. (46) of Rao, Moessner & Knolle (PRB 112,
        024440, 2025) -- at a single momentum point, either from the bare bubble (rpa=False)
        or the RPA-dressed susceptibility (rpa=True, default; this is what Figs. 1, 4, 6-13 of
        that paper plot). Reuses compute_free_susceptibility_k's raw tensor internally (no
        redundant computation) and applies susceptibility_spin_project (already in this file)
        for the polarization projection.

        Returns a (Nx_omega,) real array (the imaginary part; susceptibility_spin_project
        already returns Im via the tensor's construction, matching the paper's convention).
        """
        free_parameters = Susceptibility_parameters(
            Parameters(self.Nx, self.Ny, self.Kxyz, self.Jxyz, self.Gamma, self.epsilon),
            self.h_field, self.h_dir, self.len_chucks, self.omega_max, self.delta,
            self.N_core, Ek, uk)
        tensor = Compute_susceptibility_tensor(free_parameters, omega=None, dtype=self.dtype)
        susceptibility_0_raw = tensor.calculate_susceptibility_free_Parallel(kx, ky)[0]  # (Nx,2,3,2,3)

        if rpa:
            self.Interaction_vertices()
            susceptibility_raw = self.RPA_kernel(kx, ky, susceptibility_0_raw)
        else:
            susceptibility_raw = susceptibility_0_raw

        return self.susceptibility_spin_project(susceptibility_raw, kx, ky).imag

    def high_symmetry_path_indices(self, path_points_per_segment=None):
        """
        Standard honeycomb-lattice high-symmetry points (Gamma, K, K', and the three
        inequivalent M-type edge midpoints), derived from THIS model's own reciprocal
        lattice vectors (self.bmatrix), and assembled into a 6-segment path
        Gamma -> K -> M(a) -> K1 -> K' -> M(b) -> Gamma in the same spirit as Fig. 3(c) of
        Rao, Moessner & Knolle (PRB 112, 024440, 2025) / Fig. 1(b) of Chen & Wang
        (arXiv:2603.24918).

        DERIVATION (verified numerically): with b1, b2 = self.bmatrix[:,0], self.bmatrix[:,1],
        |b1|=|b2|=1 and the angle between them is exactly 120 degrees (a regular hexagonal
        reciprocal lattice, as it must be for a honeycomb direct lattice). The three
        inequivalent M points are half of each of the three shortest reciprocal-lattice
        vectors: M_a=b1/2, M_b=b2/2, M_c=(b1+b2)/2. The two inequivalent BZ corners are the
        standard K=(2*b1+b2)/3, K'=(b1+2*b2)/3 (verified: |K|/|M_a| = 2/sqrt(3), the exact
        ratio for a regular hexagon). "K1" is taken as a second-Brillouin-zone copy of K,
        K1 = K - b1 (matching the papers' own remark that some instabilities condense at
        "K1 points on the second BZ" rather than the first-BZ K).

        CAVEAT: this reproduces the same STANDARD high-symmetry points of the identical
        honeycomb BZ used by both papers, verified by construction (hexagon geometry, exact
        K/M ratio) -- but the exact numeric path in either paper's Fig. 3(c)/Fig. 1(b) (which
        of the three equivalent M-type points is labeled "M" vs "M1"/"M2", and the precise
        second-BZ copy chosen for K1) was not cross-checked against their original source
        code, since it is not public in machine-readable form for this session. If you need
        pixel-for-pixel agreement with a specific paper's panel, treat this as a very close
        but not guaranteed byte-identical convention.

        Requires self.Nx divisible by 6 for the K and M points to land exactly on integer
        grid indices (asserted below; use a compatible Nx, e.g. 24, 30, 60, 90...).

        Returns (path_kxky, tick_positions, tick_labels): path_kxky is an (n_points, 2) array
        of (kx, ky) INTEGER grid indices (already wrapped mod Nx) tracing the path linearly
        between each pair of consecutive high-symmetry points; tick_positions/tick_labels are
        for labeling the resulting plot's x-axis.
        """
        assert self.Nx % 6 == 0, (
            f'high_symmetry_path_indices needs self.Nx divisible by 6 (got {self.Nx}) so K '
            f'(at Nx/3) and M (at Nx/2) land exactly on integer grid points.')
        if path_points_per_segment is None:
            path_points_per_segment = max(self.Nx // 6, 4)

        Nx = self.Nx
        Gamma = np.array([0, 0])
        K = np.array([2 * Nx // 3, Nx // 3])
        Kp = np.array([Nx // 3, 2 * Nx // 3])
        Ma = np.array([Nx // 2, 0])
        Mb = np.array([0, Nx // 2])
        K1 = K - np.array([Nx, 0])  # second-BZ copy of K

        vertices = [Gamma, K, Ma, K1, Kp, Mb, Gamma]
        labels = [r'$\Gamma$', 'K', r'$M_1$', r'$K_1$', r"$K'$", 'M', r'$\Gamma$']

        path_kxky = []
        tick_positions = [0]
        for seg in range(len(vertices) - 1):
            start, end = vertices[seg], vertices[seg + 1]
            n = path_points_per_segment
            for t in range(n):
                point = start + (end - start) * t / n
                path_kxky.append(point)
            tick_positions.append(len(path_kxky))
        path_kxky.append(vertices[-1])
        path_kxky = np.array(path_kxky)
        path_kxky = np.round(path_kxky).astype(int) % Nx

        return path_kxky, tick_positions, labels

    def scan_BZ_for_soft_modes(self, Ek, uk, kx_list, ky_list, components=('xx', 'yy', 'zz'),
                               threshold=50.0):
        """
        STAGE 1 of soft-mode hunting: scan the RPA susceptibility over a list of (kx, ky)
        grid points and report, per channel, the lowest-omega bin at which each point's
        response exceeds `threshold` -- i.e. is clearly not part of the smooth two-Majorana
        continuum (whose Im chi is O(1-10) in these units; a genuine sharp mode/pole is
        typically 2-3 orders of magnitude larger even at moderate delta). This is a coarse
        first pass, meant to be cheap: use a modest self.Nx (30-40) and a not-too-fine
        self.delta for this step, then refine candidates with refine_soft_mode_location() and
        verify them with verify_pole() before trusting any result from this scan alone.

        kx_list, ky_list : same-length sequences of integer BZ indices to scan (e.g. a 1D
            path like KJGamma_MF_E_aniso.py's susceptibility_data_save(plot_type='path')
            convention, or a coarse 2D grid via np.meshgrid -- a genuine instability's
            momentum is not guaranteed to sit on a high-symmetry line; see the paper's own
            incommensurate-spiral example, Fig. 8, where it doesn't).

        Returns a dict {component: list of (kx, ky, omega, peak_value)}, one entry per scanned
        point where a candidate was found (empty for points with no bin above threshold),
        sorted by omega ascending (softest / most dangerous candidates first).
        """
        index = {'xx': 0, 'yy': 1, 'zz': 2}
        omega_axis = np.arange(self.Nx) * self.omega_step
        candidates = {comp: [] for comp in components}

        for kx, ky in zip(kx_list, ky_list):
            _, chi_rpa = self.compute_free_susceptibility_k(Ek, uk, int(kx) % self.Nx,
                                                            int(ky) % self.Ny, rpa=True)
            for comp in components:
                curve = chi_rpa[:, index[comp], index[comp]].imag
                above = np.where(curve > threshold)[0]
                if len(above):
                    i0 = above[0]
                    candidates[comp].append((int(kx) % self.Nx, int(ky) % self.Ny,
                                             omega_axis[i0], curve[i0]))

        for comp in components:
            candidates[comp].sort(key=lambda t: t[2])
        return candidates

    def verify_pole(self, Ek, uk, kx, ky, comp, delta_fractions=(0.6, 0.3, 0.15, 0.075),
                    omega_max=None):
        """
        STAGE 2: confirm a candidate found by scan_BZ_for_soft_modes() is a genuine RPA pole
        and not a coarse-grid artifact, by checking that peak_height * delta stays
        approximately CONSTANT as delta is shrunk (the textbook signature of a real resonance
        -- a broad/smooth feature's peak height would barely change as delta shrinks, while a
        true pole's height scales as ~1/delta until the frequency grid becomes too coarse to
        sample it, at which point the product starts dropping instead of staying flat).

        comp : 'xx', 'yy', or 'zz' -- checks a single channel (call again for others).
        delta_fractions : self.delta is temporarily overridden to delta_fractions[i]/self.Nx
            for each successive check; the DEFAULT sequence roughly halves delta each step.
        omega_max : optional override (defaults to self.omega_max) -- use something not much
            bigger than the candidate's own omega so you keep enough frequency resolution
            near it as delta shrinks (self.omega_step = omega_max/self.Nx is fixed by Nx
            regardless of omega_max, so a smaller omega_max gives finer sampling for free).

        Returns a list of (delta, peak_value, peak_omega, peak*delta) tuples -- eyeball
        whether the last column is roughly flat (genuine pole) or trending down after an
        initial plateau (artifact / under-resolved).
        """
        index = {'xx': 0, 'yy': 1, 'zz': 2}
        i = index[comp]
        results = []
        for frac in delta_fractions:
            sp = Susceptibility_parameters(
                Parameters(self.Nx, self.Ny, self.Kxyz, self.Jxyz, self.Gamma, self.epsilon),
                self.h_field, self.h_dir, self.len_chucks,
                omega_max if omega_max is not None else self.omega_max,
                frac / self.Nx, self.N_core, Ek, uk)
            tensor = Compute_susceptibility_tensor(sp, omega=None, dtype=self.dtype)
            raw = tensor.calculate_susceptibility_free_Parallel(kx, ky)[0]
            self.Interaction_vertices()
            raw_rpa = self.RPA_kernel(kx, ky, raw)
            chi = self.susceptibility_spin_basis(raw_rpa, kx, ky)
            curve = chi[:, i, i].imag
            idx = curve.argmax()
            omega_axis = np.arange(self.Nx) * (sp.omega_max / self.Nx)
            results.append((frac / self.Nx, curve[idx], omega_axis[idx], curve[idx] * frac / self.Nx))
        return results

    def refine_soft_mode_location(self, Ek, uk, kx0, ky0, comp, radius=2, rpa=True):
        """
        STAGE 3: local refinement around a candidate (kx0, ky0) found by
        scan_BZ_for_soft_modes() -- scans every grid point in the (2*radius+1)^2 box centered
        on it and returns the one with the lowest softening omega (the "most condensed"
        nearby point), since a coarse scan only localizes a mode to within one grid cell.

        Single-channel convenience wrapper. If you need more than one channel at the same
        (kx0, ky0), call refine_soft_mode_locations() (plural) instead -- it scans the box
        ONCE and extracts all requested channels from the same calculation, rather than
        recomputing the same susceptibility tensor once per channel as looping this method
        would.

        Returns (best_kx, best_ky, best_omega, best_peak_value).
        """
        return self.refine_soft_mode_locations(Ek, uk, kx0, ky0, (comp,), radius)[comp]

    def refine_soft_mode_locations(self, Ek, uk, kx0, ky0, components=('xx', 'yy', 'zz'), radius=2):
        """
        STAGE 3 (multi-channel, efficient): local refinement around a candidate (kx0, ky0),
        scanning every point in the (2*radius+1)^2 box ONCE per momentum point -- since
        compute_free_susceptibility_k already returns all three diagonal channels from a
        single calculation, scanning per-channel separately (calling
        refine_soft_mode_location() three times) would recompute the same susceptibility
        tensor 3x at every point for no reason.

        Returns a dict {component: (best_kx, best_ky, best_omega, best_peak)}, one entry per
        requested channel, each independently minimized over the SAME shared box scan.
        """
        index = {'xx': 0, 'yy': 1, 'zz': 2}
        omega_axis = np.arange(self.Nx) * self.omega_step
        best = {comp: None for comp in components}
        for dkx in range(-radius, radius + 1):
            for dky in range(-radius, radius + 1):
                kx, ky = (kx0 + dkx) % self.Nx, (ky0 + dky) % self.Ny
                _, chi_rpa = self.compute_free_susceptibility_k(Ek, uk, kx, ky, rpa=True)
                for comp in components:
                    curve = chi_rpa[:, index[comp], index[comp]].imag
                    idx = curve.argmax()
                    if best[comp] is None or omega_axis[idx] < best[comp][2]:
                        best[comp] = (kx, ky, omega_axis[idx], curve[idx])
        return best

    def _draw_free_susceptibility_curve(self, ax, omega_axis, curve, comp, smooth,
                                        linestyle='solid', label_suffix='', color=None):
        """
        Draw one Im chi^{comp,comp}(omega) curve onto a given Axes, optionally applying a
        COSMETIC (visualization-only, does not change the underlying computation) moving-
        average smoothing. Shared by the combined and separate-figures modes of
        plot_free_susceptibility_momentum_cut, and by both the pre-RPA and RPA-dressed curves
        (via linestyle/label_suffix/color) so all of them stay pixel-for-pixel consistent.

        smooth : None or odd int >= 3
            None -- plot the raw curve.
            odd int w -- convolve with a length-w normalized boxcar (moving average) before
                plotting; edges use 'same'-mode convolution (boundary bins average over
                fewer points, standard edge handling). This is purely a display aid for
                already-computed data: it cannot recover resolution that was never computed.
                Real smoothness comes from a finer Nx / smaller delta -- see
                plot_free_susceptibility_momentum_cut's docstring.
        """
        if smooth is not None:
            if smooth < 3 or smooth % 2 == 0:
                raise ValueError('smooth must be an odd integer >= 3')
            kernel = np.ones(smooth) / smooth
            curve = np.convolve(curve, kernel, mode='same')
        label = r'$\mathrm{Im}\,\chi_0^{%s}$%s' % (comp, label_suffix)
        ax.plot(omega_axis, curve, linewidth=1.3, linestyle=linestyle, color=color, label=label)

    def plot_free_susceptibility_momentum_cut(self, Ek, uk, kx=0, ky=0,
                                              components=('xx', 'yy', 'zz'), save='n',
                                              combined=True, smooth=None, figsize=None,
                                              rpa=False):
        """
        Fig.-4(b)-style plot: the BARE (pre-RPA) Majorana MFT susceptibility Im chi_0^{aa}
        (omega) at a fixed momentum (default k=0), one curve per requested diagonal spin
        component -- either overlaid on one set of axes (combined=True, default) or as
        SEPARATE individual figures/files, one per component (combined=False). This is the
        "before RPA" response by default -- and, like the paper's Fig. 4(b) INSET specifically
        (not its main, RPA-dressed panel), it needs no packed/saved data on disk: it calls
        compute_free_susceptibility_k() directly. Per the paper's own caption, that inset is
        exactly this same object -- chi_0 with unshifted peak onset -- evaluated at a much
        finer grid (L=240, delta=1/120) than the main text's default, to keep it smooth; see
        the "smoothness" note below.

        GENERALIZATION to anisotropic Heisenberg exchange: in the paper's pure-Kitaev
        benchmark (Fig. 4(b)), Im chi_0^{xx} = Im chi_0^{yy} = Im chi_0^{zz} by C3 symmetry,
        so only one curve (labeled "zz") is shown. Once self.Jxyz has Jx != Jy != Jz, that
        symmetry is broken and the three components genuinely differ -- this method plots
        all three (or whichever subset you request via `components`); it collapses back to
        visually identical curves automatically in the isotropic limit (verified numerically
        to converge as Nx increases).

        rpa : bool, default False
            If True, ALSO compute and overlay the RPA-DRESSED response (Eq. 25, via
            RPA_kernel -- see compute_free_susceptibility_k(rpa=True)) for every requested
            component: the pre-RPA curve is drawn solid, the RPA-dressed curve dashed, in a
            matching color, both sharing the same legend entry group per component. This is
            the direct analogue of comparing the paper's Fig. 4(b) main panel (RPA-dressed)
            against its inset (bare bubble) -- except here for a genuinely anisotropic model,
            where the RPA dressing can shift/sharpen each spin channel differently (the same
            "sharp paramagnon mode" phenomenology Sec. V of the paper discusses for nonzero
            J, Gamma, just resolved per spin component instead of assuming they're identical).
            Uses the SAME raw bare-bubble computation for both curves (no recomputation).

        SMOOTHNESS: the underlying resolution is set by self.Nx -- it is BOTH the BZ
        momentum-grid size (the q-sum in calculate_susceptibility_free_Parallel_q runs over
        self.Nx * self.Ny points) AND the frequency-grid size (self.omega_step =
        self.omega_max / self.Nx), so raggedness at small Nx is genuine under-resolution, not
        a plotting artifact. The real fix is the same one the paper uses for its inset:
        increase Nx (rebuild Ek, uk at the larger Nx via
        Majorana_rep.get_eiegensystem_Majorana() and re-solve/re-diagonalize -- cost scales
        as Nx^2 per momentum point requested, so this gets expensive quickly) and/or increase
        self.delta (more Lorentzian broadening of each pole -- cheap, but washes out genuinely
        sharp features like the van Hove peak if pushed too far, exactly the same tradeoff the
        paper describes between its main-panel and inset delta values). The `smooth` argument
        below is a purely COSMETIC, zero-cost alternative when you want a nicer-looking curve
        from data you've already computed and don't want to re-run at higher Nx -- it cannot
        recover real spectral resolution the coarser grid never had.

        Ek, uk : as in compute_free_susceptibility_k() -- must come from a Majorana_rep model
            with the SAME Nx, Ny, Kxyz, Jxyz, Gamma, epsilon, h_dir, h_field as self.
        components : subset of {'xx','yy','zz'}, default all three.
        combined : bool, default True
            True -- one figure, all requested components overlaid with a legend (as above).
            False -- one SEPARATE figure per component; save='y' writes one file per
                component (component name appended before the extension).
        smooth : None or odd int >= 3, default None
            Cosmetic moving-average window (in grid points) applied only for display -- see
            above. Try e.g. 5-9 for a modest touch-up on a coarse-Nx curve. Applied
            identically to both the pre-RPA and RPA-dressed curves when rpa=True.
        """
        index = {'xx': 0, 'yy': 1, 'zz': 2}
        for comp in components:
            if comp not in index:
                raise NotImplementedError("components must be a subset of {'xx','yy','zz'}")

        if rpa:
            chi0, chi_rpa = self.compute_free_susceptibility_k(Ek, uk, kx, ky, rpa=True)
        else:
            chi0 = self.compute_free_susceptibility_k(Ek, uk, kx, ky, rpa=False)
            chi_rpa = None
        omega_axis = np.arange(self.Nx) * self.omega_step

        if kx == 0 and ky == 0:
            momentum_label = r'\mathbf{k} = 0'
        else:
            k1, k2, k3 = np.einsum('ij,j->i', self.bmatrix, np.array([kx, ky]) / self.Nx)
            momentum_label = r'\mathbf{k} = %s' % str((k1, k2))

        param_line = (r'$K_{xyz}=%s,\ J_{xyz}=%s,\ \Gamma=%s,\ \varepsilon=%s$, $h=%s$'
                      % ([round(float(k), 4) for k in self.Kxyz], [round(float(j), 4) for j in self.Jxyz],
                         round(float(self.Gamma), 4), [round(float(e), 4) for e in self.epsilon],
                         self.h_field))
        y_label = (r'$\mathrm{Im}\,\chi_0$ (a.u.), solid: before RPA, dashed: RPA (Eq. 25)'
                   if rpa else r'$\mathrm{Im}\,\chi_0$ (a.u.), before RPA')
        title_prefix = 'MFT response, before RPA (solid) vs RPA-dressed (dashed)' if rpa \
            else 'MFT response before RPA'

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if save not in ('y', 'n'):
            raise NotImplementedError('wrong input. save = y,n  ')

        if combined:
            fig, ax = plt.subplots(figsize=figsize or (6.4, 4.8))
            for n, comp in enumerate(components):
                color = default_colors[n % len(default_colors)]
                self._draw_free_susceptibility_curve(
                    ax, omega_axis, chi0[:, index[comp], index[comp]].imag, comp, smooth,
                    linestyle='solid', label_suffix=(' (pre-RPA)' if rpa else ''), color=color)
                if rpa:
                    self._draw_free_susceptibility_curve(
                        ax, omega_axis, chi_rpa[:, index[comp], index[comp]].imag, comp, smooth,
                        linestyle='dashed', label_suffix=' (RPA)', color=color)
            ax.set_xlabel(r'$\omega\ (|K|)$')
            ax.set_ylabel(y_label)
            ax.set_xlim(0, self.omega_max)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            fig.suptitle(r'%s, $%s$' % (title_prefix, momentum_label) + '\n' + param_line, fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.90])

            if save == 'y':
                self.load_label = 'free'
                self.save_type_name()
                suffix = '_before_and_after_RPA' if rpa else '_before_RPA'
                fig.savefig(f'{self.save_path}{suffix}(cut{kx},{ky}).png', dpi=300, format='png')
            plt.show()

        else:
            for comp in components:
                fig, ax = plt.subplots(figsize=figsize or (5.2, 4.2))
                self._draw_free_susceptibility_curve(
                    ax, omega_axis, chi0[:, index[comp], index[comp]].imag, comp, smooth,
                    linestyle='solid', label_suffix=(' (pre-RPA)' if rpa else ''))
                if rpa:
                    self._draw_free_susceptibility_curve(
                        ax, omega_axis, chi_rpa[:, index[comp], index[comp]].imag, comp, smooth,
                        linestyle='dashed', label_suffix=' (RPA)')
                ax.set_xlabel(r'$\omega\ (|K|)$')
                ax.set_ylabel(y_label.replace(r'\chi_0', r'\chi_0^{%s}' % comp))
                ax.set_xlim(0, self.omega_max)
                ax.set_ylim(bottom=0)
                if rpa:
                    ax.legend(fontsize=8)
                fig.suptitle(r'%s, $%s$' % (title_prefix, momentum_label) + '\n' + param_line, fontsize=8)
                fig.tight_layout(rect=[0, 0, 1, 0.88])

                if save == 'y':
                    self.load_label = 'free'
                    self.save_type_name()
                    suffix = '_before_and_after_RPA' if rpa else '_before_RPA'
                    fig.savefig(f'{self.save_path}{suffix}_{comp}(cut{kx},{ky}).png', dpi=300, format='png')
                plt.show()

    def plot_susceptibility_BZ_path_1D(self, Ek, uk, omega_index=0, integrate=False,
                                       components=('xx', 'yy', 'zz'), rpa=False, save='n',
                                       path_points_per_segment=None, figsize=None):
        """
        1D MOMENTUM line plot: Im chi(k) plotted as a single curve vs distance along the
        standard Gamma -> K -> M1 -> K1 -> K' -> M -> Gamma high-symmetry path (see
        high_symmetry_path_indices()), collapsing the frequency axis instead of showing it
        as a color map -- the 1D-curve companion to the existing 2D path heatmaps
        (plot_susceptibility_path_1stBZ/_2ndBZ) and to plot_free_susceptibility_momentum_cut
        (which is the same idea with the roles of k and omega swapped: ONE fixed k, a curve
        vs omega).

        Like plot_free_susceptibility_momentum_cut, this needs NO packed/saved data on disk
        -- it calls compute_free_susceptibility_k() directly at every k along the path, so it
        works without ever running the separate chunk-generation pipeline that
        pack_susceptibility_projected() depends on (see that method's docstring).

        Ek, uk : as in compute_free_susceptibility_k() -- from a converged Majorana_rep model
            with the SAME Nx, Ny, Kxyz, Jxyz, Gamma, epsilon, h_dir, h_field as self.
        omega_index : int, default 0. Ignored if integrate=True. Which frequency bin (0 ..
            self.Nx-1, i.e. omega = omega_index * self.omega_step) to slice the momentum
            curve at. omega_index=0 is the closest available proxy to the true static
            (omega=0) susceptibility at this discretization -- exact omega=0 can be a pole
            of the free bubble at some k (a gapless two-Majorana continuum touching zero
            energy), so it is never evaluated directly, only approached via the first bin.
        integrate : bool, default False. If True, plot the frequency-integrated response
            integral_0^omega_max d_omega Im chi^{aa}(k,omega) (trapezoidal rule over the full
            omega grid) instead of a single-frequency slice -- overrides omega_index. This is
            an equal-time-like (static structure factor S(k)-flavored) quantity, useful when
            you want one number per k summarizing the whole spectrum rather than picking one
            frequency.
        components : subset of {'xx','yy','zz'}, default all three.
        rpa : bool, default False. If True, ALSO compute and overlay the RPA-dressed curve
            (dashed) alongside the bare one (solid), per component, exactly as in
            plot_free_susceptibility_momentum_cut. Uses the SAME raw bare-bubble computation
            for both curves (no recomputation) via compute_free_susceptibility_k(rpa=True).
        path_points_per_segment : passed through to high_symmetry_path_indices(); default
            None uses that method's own default (self.Nx // 6).

        COST NOTE: this calls compute_free_susceptibility_k() once per path point (each one
        itself an O(Nx*Ny) BZ sum internally), so total cost scales as (path length) x
        (Nx*Ny) -- this can take a while at large Nx. Lower path_points_per_segment and/or
        self.Nx for a quick look before committing to a fine grid.

        Returns (curves_0, curves_rpa): dicts keyed by component name, each value a
        length-n_points 1D array (curves_rpa is None if rpa=False).
        """
        index = {'xx': 0, 'yy': 1, 'zz': 2}
        for comp in components:
            if comp not in index:
                raise NotImplementedError("components must be a subset of {'xx','yy','zz'}")
        if save not in ('y', 'n'):
            raise NotImplementedError('wrong input. save = y,n  ')

        path_kxky, tick_positions, tick_labels = self.high_symmetry_path_indices(path_points_per_segment)
        n_points = len(path_kxky)

        curves_0 = {comp: np.zeros(n_points) for comp in components}
        curves_rpa = {comp: np.zeros(n_points) for comp in components} if rpa else None

        for i, (kx, ky) in enumerate(path_kxky):
            if rpa:
                chi0, chi_rpa = self.compute_free_susceptibility_k(Ek, uk, int(kx), int(ky), rpa=True)
            else:
                chi0 = self.compute_free_susceptibility_k(Ek, uk, int(kx), int(ky), rpa=False)
                chi_rpa = None

            for comp in components:
                a = index[comp]
                if integrate:
                    curves_0[comp][i] = np.trapz(chi0[:, a, a].imag, dx=self.omega_step)
                    if rpa:
                        curves_rpa[comp][i] = np.trapz(chi_rpa[:, a, a].imag, dx=self.omega_step)
                else:
                    curves_0[comp][i] = chi0[omega_index, a, a].imag
                    if rpa:
                        curves_rpa[comp][i] = chi_rpa[omega_index, a, a].imag

        param_line = (r'$K_{xyz}=%s,\ J_{xyz}=%s,\ \Gamma=%s,\ \varepsilon=%s$, $h=%s$'
                      % ([round(float(k), 4) for k in self.Kxyz], [round(float(j), 4) for j in self.Jxyz],
                         round(float(self.Gamma), 4), [round(float(e), 4) for e in self.epsilon],
                         self.h_field))
        if integrate:
            y_label = r'$\int d\omega\ \mathrm{Im}\,\chi_0(\mathbf{k},\omega)$ (a.u.)'
            slice_label = 'frequency-integrated'
        else:
            omega_value = omega_index * self.omega_step
            y_label = r'$\mathrm{Im}\,\chi_0(\mathbf{k},\omega=%.4f)$ (a.u.)' % omega_value
            slice_label = r'$\omega=%.4f$' % omega_value
        if rpa:
            y_label += ', solid: before RPA, dashed: RPA (Eq. 25)'
        title = (f'MFT response along ' + r'$\Gamma$-K-M$_1$-K$_1$-K$^\prime$-M-$\Gamma$'
                 + f', {slice_label}'
                 + (' (solid: before RPA, dashed: RPA)' if rpa else ' (before RPA)'))

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        path_axis = np.arange(n_points)
        fig, ax = plt.subplots(figsize=figsize or (7.2, 4.8))
        for n, comp in enumerate(components):
            color = default_colors[n % len(default_colors)]
            ax.plot(path_axis, curves_0[comp], linewidth=1.3, linestyle='solid', color=color,
                    label=r'$\mathrm{Im}\,\chi_0^{%s}$%s' % (comp, ' (pre-RPA)' if rpa else ''))
            if rpa:
                ax.plot(path_axis, curves_rpa[comp], linewidth=1.3, linestyle='dashed', color=color,
                        label=r'$\mathrm{Im}\,\chi_0^{%s}$ (RPA)' % comp)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        for tp in tick_positions[1:-1]:
            ax.axvline(tp, color='gray', linewidth=0.5, linestyle=':')
        ax.set_ylabel(y_label)
        ax.set_xlim(0, n_points - 1)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        fig.suptitle(title + '\n' + param_line, fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.90])

        if save == 'y':
            self.load_label = 'free'
            self.save_type_name()
            suffix = '_integrated' if integrate else f'_omega{omega_index}'
            rpa_suffix = '_before_and_after_RPA' if rpa else '_before_RPA'
            fig.savefig(f'{self.save_path}_1Dpath{rpa_suffix}{suffix}.png', dpi=300, format='png')
        plt.show()

        return curves_0, curves_rpa

    def plot_susceptibility_frequency_cut(self, omega, save='n'):
        assert self.data_type != 'path'
        if self.data_type == 'BZ':
            susceptibility_const = (self.susceptibility[:, :, int(omega / self.omega_step)]).imag
        else:
            susceptibility_const = (self.susceptibility[:, :]).imag

        susceptibility_array = susceptibility_const
        max_value = susceptibility_array.max()
        min_value = susceptibility_array.min()

        kvec = np.array(np.meshgrid(np.arange(-self.Nx, self.Nx, 1), np.arange(-self.Ny, self.Ny, 1)))
        kx, ky, kz = np.einsum("ij,jkl", self.bmatrix, kvec)

        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.BoundaryNorm(np.linspace(min_value, max_value, 256), ncolors=cmap.N, clip=True)
        fig, axes = plt.subplots()
        const_fig = axes.pcolormesh(kx, ky, susceptibility_array.reshape(kx.shape).T, cmap=cmap, norm=norm)

        # 1st BZ hexagon
        cc = self.Nx * np.sqrt(3) / 3 * np.exp(1j * np.pi * (1. / 3 * np.arange(7)))
        hh = Polygon(np.c_[cc.real, cc.imag],
                     lw=1, edgecolor='r',
                     clip_on=False, facecolor='none')
        axes.add_patch(hh)
        # 2nd BZ hexagon
        cc = self.Nx * np.exp(1j * np.pi * (1. / 6 + 1. / 3 * np.arange(7)))
        hh = Polygon(np.c_[cc.real, cc.imag],
                     lw=1, edgecolor='m',
                     clip_on=False, facecolor='none')
        axes.add_patch(hh)
        axes.set_aspect(1)
        cbar = plt.colorbar(const_fig, ticks=np.linspace(min_value, max_value, 5))

        cbar.ax.set_ylabel(f'$\\mathcal{{I}} \ $(a.u.)', rotation=270)
        plt.xticks([])
        plt.yticks([])

        plt.title(
            f'$\\mathcal{{I}}_{{{self.load_label}}}(\\mathbf{{k}}, \\omega = {omega}|K|), h = {self.h_field}{self.plot_title}$')
        if save == 'y':
            plt.savefig(f'{self.save_path}_projected(cut{omega}|K|).png', dpi=300, format='png')
        elif save == 'n':
            pass
        else:
            raise NotImplementedError('wrong input. save = y,n  ')
        plt.show()

    def save_type_name(self):

        if self.h_dir == 'c':
            self.plot_title = save_title = '(111)'
        elif self.h_dir == 'b':
            self.plot_title = '(1\\bar{1}0)'
            save_title = '(1bar10)'

        save_directory = (f'{os.getcwd()}/results/{save_title}/'
                          f'K{[round(float(k), 6) for k in np.atleast_1d(self.Kxyz)]}'
                          f'J{[round(float(j), 6) for j in np.atleast_1d(self.Jxyz)]}'
                          f'Gamma{round(float(self.Gamma), 6)}'
                          f'eps{[round(float(e), 6) for e in np.atleast_1d(self.epsilon)]}/')

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        self.save_path = f'{save_directory}Nx{self.Nx}h={self.h_field}_susceptibility_Majorana_{self.load_label}'


class Compute_susceptibility_tensor(object):

    def __init__(self, susceptibility_parameters, omega=None, dtype=np.complex128):
        self.parameters = susceptibility_parameters.parameters
        self.Nx = self.parameters.Nx
        self.Ny = self.parameters.Ny

        self.omega_max = susceptibility_parameters.omega_max
        self.delta = susceptibility_parameters.delta
        self.len_chucks = susceptibility_parameters.len_chunks
        self.N_core = susceptibility_parameters.N_core
        self.omega_step = self.omega_max / self.Nx

        self.h_dir = susceptibility_parameters.h_dir
        self.h_field = susceptibility_parameters.h_field

        self.Ek = susceptibility_parameters.Ek
        self.uk = susceptibility_parameters.uk
        self.omega = omega
        self.data_type = isinstance(self.omega, (int, float, complex))
        self.dtype = dtype

    def calculate_susceptibility_free_Parallel(self, kx, ky):

        susceptibility_0 = np.zeros((1, self.Nx, 2, 3, 2, 3), dtype=self.dtype)

        print("starting new k...")

        for qx in np.arange(self.Nx):
            for qy in np.arange(self.Ny):
                susceptibility_0[0] += self.calculate_susceptibility_free_Parallel_q(qx, qy, kx, ky)

        return susceptibility_0 / (self.Nx * self.Ny)

    def calculate_susceptibility_free_Parallel_q(self, qx, qy, kx, ky):

        kq_x, kq_y = np.array([(kx + qx) % self.Nx, (ky + qy) % self.Ny])
        minusq_x, minusq_y = np.array([(-qx) % self.Nx, (-qy) % self.Ny])
        minuskq_x, minuskq_y = np.array([(-kx - qx) % self.Nx, (-ky - qy) % self.Ny])

        U_eig_kq = self.uk[kq_x, kq_y]
        U_eig_q = self.uk[qx, qy]
        U_eig_minuskq = self.uk[minuskq_x, minuskq_y]
        U_eig_minusq = self.uk[minusq_x, minusq_y]

        uc_kq = np.array([U_eig_kq[4 * i] for i in np.arange(2)]).reshape(2, 4)
        ub_kq = np.array([U_eig_kq[4 * i + 1:4 * i + 4] for i in np.arange(2)]).reshape(2, 3, 4)
        uc_q = np.array([U_eig_q[4 * i] for i in np.arange(2)]).reshape(2, 4)
        ub_q = np.array([U_eig_q[4 * i + 1:4 * i + 4] for i in np.arange(2)]).reshape(2, 3, 4)

        uc_minuskq = np.array([U_eig_minuskq[4 * i] for i in np.arange(2)]).reshape(2, 4)
        ub_minuskq = np.array([U_eig_minuskq[4 * i + 1:4 * i + 4] for i in np.arange(2)]).reshape(2, 3, 4)
        uc_minusq = np.array([U_eig_minusq[4 * i] for i in np.arange(2)]).reshape(2, 4)
        ub_minusq = np.array([U_eig_minusq[4 * i + 1:4 * i + 4] for i in np.arange(2)]).reshape(2, 3, 4)

        Ek_kq = self.Ek[kq_x, kq_y]
        Ek_q = self.Ek[qx, qy]

        if self.data_type == False:
            omega_arr = np.arange(self.Nx) * self.omega_step  # shape (Nx,)
            denominator_1 = 1 / (omega_arr[:, np.newaxis, np.newaxis]
                                 - Ek_kq[np.newaxis, :, np.newaxis]
                                 - Ek_q[np.newaxis, np.newaxis, :] + 1j * self.delta)
            denominator_2 = 1 / (omega_arr[:, np.newaxis, np.newaxis]
                                 + Ek_kq[np.newaxis, :, np.newaxis]
                                 + Ek_q[np.newaxis, np.newaxis, :] + 1j * self.delta)
        elif self.data_type == True:
            denominator_1 = (1 / (self.omega - Ek_kq[:, np.newaxis] - Ek_q[np.newaxis, :]
                                  + 1j * self.delta))[np.newaxis, :, :]
            denominator_2 = (1 / (self.omega + Ek_kq[:, np.newaxis] + Ek_q[np.newaxis, :]
                                  + 1j * self.delta))[np.newaxis, :, :]
        else:
            raise NotImplementedError('omega input incorrect.')

        susceptibility_0 = -4 * (self.Matsubara_sum_cc_bb(uc_kq, ub_minusq, denominator_1) - self.Matsubara_sum_cc_bb(
            uc_minuskq.conj(), ub_q.conj(), denominator_2) \
                                 - self.Matsubara_sum_cb_cb(uc_kq, ub_kq, uc_minusq, ub_minusq, denominator_1) \
                                 + self.Matsubara_sum_cb_cb(uc_minuskq.conj(), ub_minuskq.conj(), uc_q.conj(),
                                                            ub_q.conj(), denominator_2))

        return susceptibility_0

    def Matsubara_sum_cc_bb(self, uc_kq, ub_minusq, denominator):

        vertex = np.einsum('in,jn,wnm -> wijm', uc_kq, uc_kq.conj(), denominator)

        return np.einsum('wijm,iam,jbm-> wiajb', vertex, ub_minusq, ub_minusq.conj())

    def Matsubara_sum_cb_cb(self, uc_kq, ub_kq, uc_minusq, ub_minusq, denominator):

        vertex = np.einsum('in,jbn,wnm->wijbm', uc_kq, ub_kq.conj(), denominator)

        return np.einsum('wijbm,jm,iam-> wiajb', vertex, uc_minusq.conj(), ub_minusq)


def load_susceptibility_data(parameters, h_field, h_dir, rep='PF'):
    Nx = parameters.Nx
    Ny = parameters.Ny
    Kxyz = parameters.Kxyz
    Gamma = parameters.Gamma
    Jxyz = list(parameters.Jxyz)
    epsilon = list(parameters.epsilon)

    if h_dir == 'c':
        load_path = f'{LOCAL_DATA_DIR}/susceptibility_data_{rep}repNx{Nx}K{Kxyz[0]}J{Jxyz}Gamma{Gamma}eps{epsilon}h(111){h_field}).pickle'
        h = np.array([1, 1, 1]) * h_field

    elif h_dir == 'b':
        load_path = f'{LOCAL_DATA_DIR}/susceptibility_data_{rep}repNx{Nx}K{Kxyz[0]}J{Jxyz}Gamma{Gamma}eps{epsilon}h(1bar10){h_field}).pickle'
        h = np.array([1, -1, 0]) * h_field
    else:
        raise NotImplementedError('wrong field directions')

    with open(load_path, 'rb') as file:
        func = pickle.load(file)

    assert Nx == func.parameters.Nx
    assert Ny == func.parameters.Ny
    assert h_field == func.h_field
    assert h_dir == func.h_dir

    Ek = func.Ek
    uk = func.uk

    return Ek, uk


@dataclass
class Susceptibility_parameters:
    parameters: Parameters
    h_field: float
    h_dir: str
    len_chunks: int
    omega_max: float
    delta: float
    N_core: int
    Ek: np.ndarray
    uk: np.ndarray


def scan_XXZ_for_M_point_softening(scan_result, threshold=50.0, omega_max=3.0,
                                   delta_val=None, N_core=1, len_chunks_val=None,
                                   components=('xx', 'yy', 'zz'), U=1, verbose=True):
    """
    Diagnostic for the known blind spot in KJGamma_MF_E_aniso.py's zero-field XXZ
    phase-diagram machinery (see that module's docstring note above build_phase_diagram_2D):
    Majorana_rep's Hamiltonian has a 2-site unit cell and CANNOT self-consistently converge
    to zigzag or stripy order, both of which require a magnetic unit cell doubled relative to
    the crystallographic cell and condense at one of the honeycomb Brillouin zone's three
    inequivalent M points. What this function CAN do, using the ALREADY-VALIDATED RPA
    susceptibility / soft-mode pipeline elsewhere in this file (scan_BZ_for_soft_modes,
    unchanged), is check -- at every point continuation_scan()/scan_XXZ_1D() labeled 'QSL' --
    whether the RPA susceptibility is softening at an M point. A genuine, growing softening
    there, coexisting with a mean-field 'QSL' label, is the standard RPA-level signature that
    the true ground state at that point is zigzag/stripy order this ansatz cannot represent --
    NOT genuine evidence for a spin liquid. Treat any flagged point's 'QSL' label as wrong
    rather than as a real phase-diagram result.

    This is a COARSE, first-pass scan (threshold-based, single k-point per M point, no pole
    verification) -- meant to tell you WHERE on a scan_result to look more closely, not to be
    the final word. For a point this flags, follow up with verify_pole() and
    refine_soft_mode_location() (both already in this file, unchanged) at that specific
    (Delta, K, J_perp, Gamma) point before reporting it as a confirmed instability.

    scan_result : output of KJGamma_MF_E_aniso.continuation_scan() / scan_XXZ_1D() -- each
        entry's 'Nx', 'Ny', 'h_dir', 'K', 'J_perp', 'Gamma', 'delta', and 'winner' (with its
        'pattern' and 'MF_parameters') fields are used directly; only entries whose winner is
        labeled 'QSL' are checked (any point already reporting nonzero magnetization is, by
        construction, not the case this function is meant to catch).
    threshold, components : passed to scan_BZ_for_soft_modes() -- see its own docstring.
    omega_max, delta_val, N_core, len_chunks_val, U : susceptibility-pipeline parameters (see
        Susceptibility_parameters/Susceptibility_evaluation); delta_val=None defaults to
        1/Nx (matching this file's default_run_parameters() convention) and len_chunks_val=
        None defaults to Nx*Ny (a single chunk covering the whole BZ -- fine for the few
        individual k-points this function actually evaluates, since
        compute_free_susceptibility_k()/scan_BZ_for_soft_modes() never read len_chunks off
        self.len_chucks for anything other than object construction here).

    Returns a list of flagged-point dicts: {'delta','K','J_perp','Gamma','M_point',
    'component','omega','peak'} -- empty if nothing was flagged (which does NOT prove the
    scan is free of an unrepresented zigzag/stripy region, only that this coarse check didn't
    catch one; see the docstring above).
    """
    # Local import (not at module level): KJGamma_MF_E_aniso.py already imports
    # (len_chunks, LOCAL_DATA_DIR) FROM this file at ITS module level, so importing back from
    # it here at module scope would be a circular import. A function-local import breaks the
    # cycle since by the time this function is actually CALLED, both modules have finished
    # loading.
    from KJGamma_MF_E_aniso import Majorana_rep

    flagged = []
    for pt in scan_result:
        winner = pt['winner']
        if winner['pattern'] != 'QSL':
            continue  # only the m=0 branch is the one this ansatz might be mislabeling

        Nx, Ny, h_dir = pt['Nx'], pt['Ny'], pt['h_dir']
        if Nx % 6 != 0 or Ny % 6 != 0:
            if verbose:
                print("  skipping Delta=%s: Nx=%d/Ny=%d not both divisible by 6 (M-point "
                      "grid indices need this -- see high_symmetry_path_indices())"
                      % (pt['delta'], Nx, Ny))
            continue

        Kxyz = np.array([pt['K']] * 3)
        Jxyz = np.array([pt['J_perp'], pt['J_perp'], pt['J_perp'] * pt['delta']])
        parameters = Parameters(Nx, Ny, Kxyz, Jxyz, pt['Gamma'], np.zeros(3))
        model = Majorana_rep(parameters, h_dir, h_field=0,
                             MF_parameters_initial=np.array(winner['MF_parameters']),
                             force_general=True)
        Ek, uk = model.get_eiegensystem_Majorana()

        d = delta_val if delta_val is not None else 1.0 / Nx
        lc = len_chunks_val if len_chunks_val is not None else Nx * Ny
        susceptibility_parameters = Susceptibility_parameters(
            parameters, 0, h_dir, lc, omega_max, d, N_core, Ek, uk)
        evaluator = Susceptibility_evaluation(susceptibility_parameters, data_type='BZ', U=U)

        # The three inequivalent M points, in the SAME (Nx, Ny) integer-grid convention as
        # high_symmetry_path_indices() elsewhere in this file (M_a = b1/2, M_b = b2/2,
        # M_c = (b1+b2)/2 -- see that method's docstring for the derivation).
        Ma = (Nx // 2, 0)
        Mb = (0, Ny // 2)
        Mc = (Nx // 2, Ny // 2)
        M_points = {'M_a': Ma, 'M_b': Mb, 'M_c': Mc}

        for label, (mkx, mky) in M_points.items():
            candidates = evaluator.scan_BZ_for_soft_modes(
                Ek, uk, [mkx], [mky], components=components, threshold=threshold)
            for comp, hits in candidates.items():
                if hits:
                    kx, ky, omega0, peak = hits[0]
                    if verbose:
                        print("  Delta=%+.3f K=%.3f J_perp=%.4f Gamma=%.4f: 'QSL' branch "
                              "but %s softens at %s (omega=%.4f, peak=%.1f) -- likely "
                              "mislabeled; true ground state is probably zigzag/stripy "
                              "order this ansatz can't represent."
                              % (pt['delta'], pt['K'], pt['J_perp'], pt['Gamma'], comp,
                                 label, omega0, peak))
                    flagged.append({'delta': pt['delta'], 'K': pt['K'], 'J_perp': pt['J_perp'],
                                    'Gamma': pt['Gamma'], 'M_point': label, 'component': comp,
                                    'omega': omega0, 'peak': peak})

    return flagged


def default_run_parameters():
    """
    Single source of truth for the default (Nx, Ny, Kxyz, Jxyz, Gamma, epsilon, N_core,
    h_dir, h_field, omega_max, delta, len_chunks) used across THIS file's initialise()
    (consumed by susceptibility_parallel.py / susceptibility_path.py for the actual raw
    BZ computation, Stage 2) and this file's own __main__ (consumed for packing/plotting
    already-computed data, Stage 3+). Both used to redeclare these independently -- which is
    exactly how Nx drifted out of sync (180 in one copy, 60 in another after an edit only
    reached one of them) and produced a confusing FileNotFoundError where the packing stage
    silently went looking for data at the wrong Nx, with nothing in the traceback pointing at
    "there are multiple copies of this number and you only changed one." Edit values ONLY
    here now -- delta is derived from Nx (1/Nx) rather than hardcoded separately, so it always
    tracks whatever Nx is set to.

    IMPORTANT: also keep KJGamma_MF_E_aniso.py's __main__ Nx/Ny/Kxyz/Jxyz/Gamma/epsilon in
    sync with these BY HAND -- that is a SEPARATE script (Stage 1, the MFT self-consistent
    solve + eigensystem save) with no dynamic import link back to this one, so this function
    cannot enforce that half of the agreement automatically. If Stage 1 and Stage 2/3 ever
    disagree on Nx again, that is the first thing to check.
    """
    Nx = Ny = 60
    Kxyz = np.array([1, 1, 1]) * 1
    Jxyz = np.array([1, 1, 1]) * (-0.09)  # (Jx, Jy, Jz): set anisotropic values to depart from the isotropic-J model
    Gamma = 0
    epsilon = np.array([0.0, 0.0, 0.0])  # (eps_x, eps_y, eps_z): electric-field couplings; 0 recovers no-E-field model
    N_core = 4
    h_dir = 'b'
    h_field = 0
    omega_max = 0.5
    len_chunks = 60
    delta = 1 / Nx
    return Nx, Ny, Kxyz, Jxyz, Gamma, epsilon, N_core, h_dir, h_field, omega_max, delta, len_chunks


# Module-level len_chunks, kept for backward compatibility with
# `from susceptibility_Majorana import len_chunks` (used by KJGamma_MF_E_aniso.py and
# run_stage2.py) -- sourced from default_run_parameters() itself so it can never drift out of
# sync with the value initialise()/__main__ actually use.
len_chunks = default_run_parameters()[-1]


def initialise():
    Nx, Ny, Kxyz, Jxyz, Gamma, epsilon, N_core, h_dir, h_field, omega_max, delta, len_chunks = \
        default_run_parameters()
    omega = None

    parameters = Parameters(Nx, Ny, Kxyz, Jxyz, Gamma, epsilon)
    Ek, uk = load_susceptibility_data(parameters, h_field, h_dir, rep='Majorana')
    susceptibility_parameters = Susceptibility_parameters(parameters, h_field, h_dir, len_chunks, omega_max, delta,
                                                          N_core, Ek, uk)

    return Compute_susceptibility_tensor(susceptibility_parameters, omega)


if __name__ == "__main__":
    # === 1D susceptibility plot: Im chi(omega) at fixed k, before RPA (solid) vs RPA-dressed
    # (dashed), one curve per spin component -- the Fig.-4(b)-style plot, generalized to
    # (possibly anisotropic-J) couplings. Uses THE SAME (Nx, Kxyz, Jxyz, Gamma, epsilon,
    # h_dir, h_field, omega_max, delta) as default_run_parameters() -- i.e. whatever you've
    # set there is the one authoritative parameter set for this script, no separate "_free"
    # copy that can silently drift out of sync with it.
    #
    # This calls compute_free_susceptibility_k() directly on the in-memory MFT eigensystem
    # (via plot_free_susceptibility_momentum_cut) -- it needs NO packed/saved data on disk,
    # unlike pack_susceptibility_projected()/load_susceptibility() below (commented out),
    # which require a SEPARATE chunk-generation script (susceptibility_parallel.py-style,
    # not part of this file) to have already run and written
    # susceptibility_Majorana_00.npy etc. for these exact parameters. If you don't have that
    # chunk-generation step, use this block; the pack_susceptibility_projected path is left
    # below, commented out, for when you do.
    Nx, Ny, Kxyz, Jxyz, Gamma, epsilon, N_core, h_dir, h_field, omega_max, delta, len_chunks = \
        default_run_parameters()

    from KJGamma_MF_E_aniso import Majorana_rep

    parameters = Parameters(Nx, Ny, Kxyz, Jxyz, Gamma, epsilon)
    # force_general=True: always solve the fully general 27-parameter (sublattice-resolved
    # m_A != m_B) mean-field ansatz, regardless of whether Kxyz/Jxyz/epsilon are symmetric
    # enough for _ansatz_size() to otherwise pick the legacy 9-/15-parameter one (which
    # forces m_A = m_B by construction) -- see KJGamma_MF_E_aniso.py's Majorana_rep
    # constructor docstring.
    model = Majorana_rep(parameters, h_dir, h_field=h_field, MF_parameters_initial=None,
                         force_general=True)
    model.warm_start_from_pure_Kitaev(seed=0)
    model.MF_parameters = model.Self_Consistent_Solution()
    model.Self_Consistent_Solution_announce()
    Ek, uk = model.get_eiegensystem_Majorana()

    susceptibility_parameters = Susceptibility_parameters(
        parameters, h_field, h_dir, len_chunks, omega_max, delta, N_core, Ek=None, uk=None)
    evaluation = Susceptibility_evaluation(susceptibility_parameters, data_type='BZ', U=1)

    # combined=True: all requested components on ONE set of axes with a legend (solid =
    # before RPA, dashed = RPA-dressed) -- set combined=False for separate figures/files,
    # one per component, instead.
    evaluation.plot_free_susceptibility_momentum_cut(
        Ek, uk, kx=0, ky=0, components=('xx', 'yy', 'zz'), save='y', combined=True, rpa=True)

    # 1D MOMENTUM-line companion: Im chi(k) along the Gamma-K-M1-K1-K'-M-Gamma path, at the
    # lowest computed frequency bin (omega_index=0, closest available proxy to static
    # omega->0) -- also in-memory, no disk pipeline needed. Uncomment to also produce this:
    # evaluation.plot_susceptibility_BZ_path_1D(
    #     Ek, uk, omega_index=0, components=('xx', 'yy', 'zz'), save='y', rpa=True)

    # === Disk-pipeline path (commented out): only works if a separate chunk-generation
    # script already wrote susceptibility_Majorana_00.npy etc. for these exact
    # (Nx, Kxyz, Jxyz, Gamma, epsilon, h_dir, h_field, omega_max, delta) -- see
    # pack_susceptibility_projected()'s docstring. Uncomment only once that data exists.
    # susceptibility_parameters_disk = Susceptibility_parameters(
    #     parameters, h_field, h_dir, len_chunks, omega_max, delta, None, None, None)
    # evaluation_disk = Susceptibility_evaluation(susceptibility_parameters_disk, data_type='BZ', U=1)
    # evaluation_disk.pack_susceptibility_projected(pack_type='RPA')
    # evaluation_disk.load_susceptibility('RPA')
    # evaluation_disk.plot_susceptibility_frequency_cut(omega=0.25, save='n')