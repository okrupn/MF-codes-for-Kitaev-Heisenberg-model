import sys
import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy import optimize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from dataclasses import dataclass



@dataclass
class Parameters:
    """
    Parameters of the extended Kitaev-Heisenberg-Gamma model in magnetic AND electric
    fields, cf. kitaev-electric-13_07_2026.tex, Eqs. (KHG), (H_E), (H_gen).

    Kxyz   : array-like, shape (3,)  -- bond-dependent Kitaev couplings (Kx, Ky, Kz)
    Jxyz   : array-like, shape (3,)  -- anisotropic Heisenberg couplings (Jx, Jy, Jz).
             On a gamma-bond the Hamiltonian carries (K_gamma + J^gamma) S^gamma S^gamma
             + sum_{beta != gamma} J^beta S^beta S^beta, cf. Eq. (HammMaj3).
    Gamma  : float                   -- off-diagonal (symmetric) Gamma coupling
    epsilon: array-like, shape (3,)  -- electric-field-induced couplings (eps_x, eps_y, eps_z).
             eps_gamma is the scalar entering the antisymmetric part of the effective
             off-diagonal coupling on the gamma-bond,
                 Lambda_gamma^{alpha beta} = Gamma + eps_gamma * epsilon^{alpha beta gamma},
             cf. Eq. (H_E) and the J_<ij>_gamma matrices in Eq. (H_gen). epsilon^{alpha beta gamma}
             is the Levi-Civita symbol, so Lambda_gamma^{alpha beta} = Gamma + eps_gamma when
             (alpha,beta,gamma) is a cyclic permutation of (x,y,z), and Gamma - eps_gamma otherwise.
             Set epsilon = (0, 0, 0) to recover the pure magnetic-field KJGamma model.
    """
    Nx: int
    Ny: int
    Kxyz: float
    Jxyz: float
    Gamma: float
    epsilon: float = None

    def __post_init__(self):
        import numpy as _np
        if self.epsilon is None:
            self.epsilon = _np.zeros(3)
        else:
            self.epsilon = _np.asarray(self.epsilon, dtype=float)
        self.Jxyz = _np.asarray(self.Jxyz, dtype=float)
        # NOTE: Kxyz was previously not validated at all (unlike Jxyz/epsilon below), which
        # meant a scalar Kxyz would silently pass __post_init__ and only fail much later, with
        # a confusing error, the first time `Kx, Ky, Kz = self.Kxyz` was unpacked inside a
        # Hamiltonian-building routine. Validate it the same way as Jxyz/epsilon.
        self.Kxyz = _np.asarray(self.Kxyz, dtype=float)
        if self.Kxyz.shape != (3,):
            raise ValueError('Kxyz must be an array-like of 3 bond-dependent Kitaev couplings (Kx, Ky, Kz)')
        if self.Jxyz.shape != (3,):
            raise ValueError('Jxyz must be an array-like of 3 anisotropic Heisenberg couplings (Jx, Jy, Jz)')
        if self.epsilon.shape != (3,):
            raise ValueError('epsilon must be an array-like of 3 electric-field couplings (eps_x, eps_y, eps_z)')


@dataclass
class Susceptibility_data:
    parameters: Parameters
    h_field: float
    h_dir: str
    len_chucks: int
    Ek: np.ndarray
    uk: np.ndarray
    vk: np.ndarray

@dataclass
class Susceptibility_data_Majorana:
    parameters: Parameters
    h_field: float
    h_dir: str
    len_chucks: int
    Ek: np.ndarray
    uk: np.ndarray
