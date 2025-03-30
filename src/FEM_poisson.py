import numpy as np
import scipy as sp
from typing import Callable

import FEM_assemble


def impose_dirichlet(A, F, a, b):
    """u(0) = a, u(1) = b"""
    A[0, :] = 0
    A[-1, :] = 0
    A[0, 0] = 1
    A[-1, -1] = 1

    F[0] = a
    F[-1] = b


def impose_neumann_dirichlet(A, F, a, b):
    """u'(0) = a, u(1) = b"""
    F[0] -= a

    A[-1, :] = 0
    A[-1, -1] = 1

    F[-1] = b


def impose_dirichlet_neumann(A, F, a, b):
    """u(0) = a, u'(1) = b"""
    F[-1] += b

    A[0, :] = 0
    A[0, 0] = 1

    F[0] = a


def solve_Poisson_dirichlet(
        f: Callable, a: float, b: float, partition: np.ndarray, deg: int
) -> np.ndarray:
    # Assemble matrices
    A = FEM_assemble.assemble_stiffness_matrix(partition, deg=deg)
    F = FEM_assemble.assemble_load_vector(partition, f, deg=deg)

    impose_dirichlet(A, F, a=a, b=b)

    # Solve the equation
    u_h = sp.sparse.linalg.spsolve(A.tocsr(), F)

    # todo: skal vi konstruere en funksjon her, eller bare returnere koordinatene?
    return u_h


if __name__ == "__main__":
    pass

