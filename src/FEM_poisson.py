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
    f: Callable, a: float, b: float, partition: np.ndarray
) -> np.ndarray:
    """Solve the 1D Poisson equation with dirichlet boundary conditions: `-u_xx = f(x), u(0)=a, u(1)=b`.

    Args:
        f (Callable): RHS of the Poisson equation.
        a (float): u(0) = a
        b (float): u(1) = b
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.

    Returns:
        np.ndarray: solution vector.
    """
    # Assemble matrices
    A = FEM_assemble.assemble_stiffness_matrix(partition)
    F = FEM_assemble.assemble_load_vector(partition, f)

    impose_dirichlet(A, F, a=a, b=b)

    # Solve the equation
    u_h = sp.sparse.linalg.spsolve(A, F)

    # todo: skal vi konstruere en funksjon her, eller bare returnere koordinatene?
    return u_h


if __name__ == "__main__":
    pass