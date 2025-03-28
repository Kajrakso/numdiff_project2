import numpy as np
import scipy as sp
import sympy as sy
import FEM_elements
from typing import Callable, Any


def get_element_load_vector(f: Callable, element: np.ndarray) -> np.ndarray:
    """Compute the elemental load vector by using the Simpsons rule.

    Args:
        f (Callable): function
        element (np.ndarray): element

    Returns:
        np.ndarray: elemental load vector
    """

    xi0 = 0
    xi1 = 1 / 2
    xi2 = 1

    x0 = FEM_elements.reference_element_to_physical_element(xi0, element)
    x1 = FEM_elements.reference_element_to_physical_element(xi1, element)
    x2 = FEM_elements.reference_element_to_physical_element(xi2, element)

    psi = FEM_elements.psi

    return (
        1
        / 6
        * np.array(
            [
                (f(x0) * psi[0](xi0) + 4 * f(x1) * psi[0](xi1) + f(x2) * psi[0](xi2)),
                (f(x0) * psi[1](xi0) + 4 * f(x1) * psi[1](xi1) + f(x2) * psi[1](xi2)),
                (f(x0) * psi[2](xi0) + 4 * f(x1) * psi[2](xi1) + f(x2) * psi[2](xi2)),
            ]
        )
    )


def assemble_load_vector(partition: np.ndarray, f: Callable) -> np.ndarray:
    """Constructs the load vector given a partition and the function f,
    the RHS of the Poisson equation: -u_xx = f(x).

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        f (Callable): function

    Returns:
        np.ndarray: load vector
    """
    M = len(partition) - 1
    N = 2 * M + 1
    F = np.zeros(N)

    elements = FEM_elements.get_elements(partition)
    element_sizes = FEM_elements.get_element_sizes(partition)

    for k in range(M):
        F_k = get_element_load_vector(f, elements[k])
        F[FEM_elements.local_to_global(k, 0) : FEM_elements.local_to_global(k + 1, 0) + 1] += (
            F_k * element_sizes[k]
        )

    return F


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


if __name__ == "__main__":
    pass
