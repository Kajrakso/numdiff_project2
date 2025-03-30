import numpy as np
import scipy as sp
from typing import Callable

import FEM_elements
from FEM_elements import local_to_global as theta
from FEM_elements import reference_element_to_physical_element as Phi_k


def construct_elemental_mass_matrix(deg: int) -> np.ndarray:
    Psis = FEM_elements.construct_Psi(deg)

    B_k = np.zeros((deg + 1, deg + 1))
    for i in range(deg + 1):
        for j in range(i, deg + 1):
            integral, _ = sp.integrate.fixed_quad(
                lambda x: Psis[i](x)*Psis[j](x), 0, 1, n=(deg + 1)
            )
            B_k[i, j] = integral
            B_k[j, i] = integral

    return B_k


def construct_elemental_stiffness_matrix(deg: int) -> np.ndarray:
    dPsis = FEM_elements.construct_dPsi(deg)

    B_k = np.zeros((deg + 1, deg + 1))
    for i in range(deg + 1):
        for j in range(i, deg + 1):
            integral, _ = sp.integrate.fixed_quad(
                lambda x: dPsis[i](x)*dPsis[j](x), 0, 1, n=deg
            )
            B_k[i, j] = integral
            B_k[j, i] = integral

    return B_k


def assemble_mass_matrix(partition: np.ndarray, deg: int) -> np.ndarray:
    """Assembles the mass matrix.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        deg (int): degree of the basis polynomials

    Returns:
        np.ndarray: mass matrix
    """
    M = len(partition) - 1          # number of segments
    N = deg * M + 1                 # number of nodes

    hk = FEM_elements.get_element_sizes(partition)

    # elemental mass matrix
    F_k = construct_elemental_mass_matrix(deg=deg)

    # build the diagonals of the sparse matrix
    diags = [np.zeros(N - i) for i in range(deg + 1)]
    offsets = np.arange(deg + 1)

    # prepare array of indices
    idxs = theta(np.arange(M)[:, None], offsets, deg=deg)

    # accumulated addition for each of the diagonals
    for i, _diag in enumerate(diags):
        np.add.at(_diag, idxs[:, :(deg + 1 - i)], np.diag(F_k, k=i) * hk[:, None])

    # assemble the upper diagonals!
    F = sp.sparse.diags_array(diags, offsets=offsets).tolil()

    # make F symmetric!
    rows, cols = F.nonzero()
    F[cols, rows] = F[rows, cols]

    return F.tolil()


def assemble_stiffness_matrix(partition: np.ndarray, deg: int) -> np.ndarray:
    """Assembles the stiffness matrix.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        deg (int): degree of the basis polynomials

    Returns:
        np.ndarray: stiffness matrix
    """
    M = len(partition) - 1          # number of segments
    N = deg * M + 1                 # number of nodes

    hk = FEM_elements.get_element_sizes(partition)

    # elemental stiffness matrix
    B_k = construct_elemental_stiffness_matrix(deg=deg)

    # prepare the diagonals of the sparse matrix
    diags = [np.zeros(N - i) for i in range(deg + 1)]
    offsets = np.arange(deg + 1)

    # prepare array of indices
    idxs = theta(np.arange(M)[:, None], offsets, deg=deg)

    # accumulated addition for each of the diagonals
    for i, _diag in enumerate(diags):
        np.add.at(_diag, idxs[:, :(deg + 1 - i)], np.diag(B_k, k=i) / hk[:, None])

    # assemble the upper diagonals!
    B = sp.sparse.diags_array(diags, offsets=offsets).tolil()

    # make B symmetric!
    rows, cols = B.nonzero()
    B[cols, rows] = B[rows, cols]

    return B.tolil()


# TODO: Cool function but not needed. Delete it!
def get_element_load_vector_fancy_but_slow(f, element):
    # vector of Psis: [Psi_0, Psi_1, Psi_2]
    Psis = FEM_elements.construct_Psi(2)

    # nodes on ref. element: [xi_0, xi_1, xi_2]
    xi = np.array([0, 1/2, 1])

    # custom outer product evaluation function
    eval_outer = lambda f, x: \
        np.frompyfunc(lambda _f, _x: _f(_x), 2, 1).outer(f, x).astype(float)

    return eval_outer(Psis, xi) @ np.diag([1/6, 2/3, 1/6]) @ f(Phi_k(xi, element))


def assemble_load_vector(partition: np.ndarray, f: Callable, deg: int) -> np.ndarray:
    """Constructs the load vector given a partition and f,
    the RHS of the Poisson equation: -u_xx = f(x).

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        f (Callable): function
        deg (int): degree of the basis polynomials

    Returns:
        np.ndarray: load vector
    """
    M = len(partition) - 1          # number of segments
    N = deg * M + 1                 # number of nodes

    # prepare the load vector
    F = np.zeros(N)

    elements = FEM_elements.get_elements(partition)
    hk = FEM_elements.get_element_sizes(partition)

    Psi = FEM_elements.construct_Psi(deg=deg)

    # integration nodes for Simpsons rule
    xi = FEM_elements.get_nodes(np.array([0, 1]), deg=deg)

    def integrand(xi, i):
        return f(Phi_k(xi, elements)) * Psi[i](xi)

    # elemental load vectors
    F_ks = np.array([
        sp.integrate.simpson(integrand(xi, i), xi) for i in range(deg + 1)
    ]).T

    # array of indices
    idxs = theta(np.arange(M)[:, None], np.arange(deg + 1), deg=deg)

    # accumulated addition
    np.add.at(F, idxs, F_ks * hk[:, None])

    return F


if __name__ == "__main__":
    """
    np.set_printoptions(precision=2)
    print("degree 2")
    part = np.linspace(0, 1, 3)
    deg = 2
    print(assemble_mass_matrix(part, deg=deg).toarray())
    print(assemble_stiffness_matrix(part, deg=deg).toarray())
    """

    M = 10
    deg = 2
    print(assemble_load_vector(np.linspace(0, 1, M+1), lambda x: x**2 + 1, deg=deg))
    # print(assemble_load_vector_old(np.linspace(0, 1, 4), lambda x: x**2 + 1))

    """
    deg = 2
    Psis = FEM_elements.construct_Psi(deg)
    print(1, sp.integrate.fixed_quad(lambda x: Psis[0](x)*Psis[0](x), 0, 1, n=1))
    print(2, sp.integrate.fixed_quad(lambda x: Psis[0](x)*Psis[0](x), 0, 1, n=2))
    print(3, sp.integrate.fixed_quad(lambda x: Psis[0](x)*Psis[0](x), 0, 1, n=3))
    print(4, sp.integrate.fixed_quad(lambda x: Psis[0](x)*Psis[0](x), 0, 1, n=4))
    print(5, sp.integrate.fixed_quad(lambda x: Psis[0](x)*Psis[0](x), 0, 1, n=5))
    print(6, sp.integrate.fixed_quad(lambda x: Psis[0](x)*Psis[0](x), 0, 1, n=6))
    print(7, sp.integrate.fixed_quad(lambda x: Psis[0](x)*Psis[0](x), 0, 1, n=7))
    """
