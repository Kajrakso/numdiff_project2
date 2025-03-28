import numpy as np
import scipy as sp
from typing import Callable
import FEM_elements
from FEM_elements import local_to_global as theta
from FEM_elements import reference_element_to_physical_element as Phi_k
from FEM_elements import Psi


def assemble_mass_matrix(partition: np.ndarray, remove_boundary=False) -> np.ndarray:
    """Assembles the mass matrix.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        remove_boundary (bool, optional): If True, the first and last rows and columns are removes from the matrix. Defaults to False.

    Returns:
        np.ndarray: mass matrix
    """
    M = len(partition) - 1
    N = 2 * M + 1
    
    hk = FEM_elements.get_element_sizes(partition)

    # elemental mass matrix
    F_k = 1 / 30 * np.array([
        [4, 2, -1],
        [2, 16, 2],
        [-1, 2, 4],
    ])

    # build the diagonals of the sparse matrix
    diag_0 = np.zeros(N)
    diag_1 = np.zeros(N - 1)
    diag_2 = np.zeros(N - 2)

    global_indices = np.array(
        [theta(k, np.array([0, 1, 2])) for k in range(M)]
    )

    # accumulated addition for each of the diagonals
    np.add.at(diag_0, global_indices[:, :3], np.diag(F_k, k=0) * hk[:, None])
    np.add.at(diag_1, global_indices[:, :2], np.diag(F_k, k=1) * hk[:, None])
    np.add.at(diag_2, global_indices[:, :1], np.diag(F_k, k=2) * hk[:, None])
    
    if remove_boundary:
        diag_0 = diag_0[1:-1]
        diag_1 = diag_1[1:-1]
        diag_2 = diag_2[1:-1]

    return sp.sparse.diags_array(
        [diag_2, diag_1, diag_0, diag_1, diag_2], offsets=[-2, -1, 0, 1, 2]
    ).tocsr()


def assemble_stiffness_matrix(
    partition: np.ndarray, remove_boundary=False
) -> np.ndarray:
    """Assembles the stiffness matrix.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        remove_boundary (bool, optional): If True, the first and last rows and columns are removed from the matrix. Defaults to False.

    Returns:
        np.ndarray: stiffness matrix
    """
    M = len(partition) - 1
    N = 2 * M + 1

    hk = FEM_elements.get_element_sizes(partition)

    # elemental stiffness matrix
    B_k = 1 / 3 * np.array([
        [7, -8, 1],
        [-8, 16, -8],
        [1, -8, 7],
    ])

    # build the diagonals of the sparse matrix
    diag_0 = np.zeros(N)
    diag_1 = np.zeros(N - 1)
    diag_2 = np.zeros(N - 2)

    global_indices = np.array(
        [theta(k, np.array([0, 1, 2])) for k in range(M)]
    )

    # accumulated addition for each of the diagonals
    np.add.at(diag_0, global_indices[:, :3], np.diag(B_k, k=0) / hk[:, None])
    np.add.at(diag_1, global_indices[:, :2], np.diag(B_k, k=1) / hk[:, None])
    np.add.at(diag_2, global_indices[:, :1], np.diag(B_k, k=2) / hk[:, None])

    if remove_boundary:
        diag_0 = diag_0[1:-1]
        diag_1 = diag_1[1:-1]
        diag_2 = diag_2[1:-1]

    return sp.sparse.diags_array(
        [diag_2, diag_1, diag_0, diag_1, diag_2], offsets=[-2, -1, 0, 1, 2]
    ).tocsr()


# TODO: Cool function but not needed. Delete it!
def get_element_load_vector_fancy_but_slow(f, element):
    # vector of Psis: [Psi_0, Psi_1, Psi_2]
    Psis = np.array(Psi)

    # nodes on ref. element: [xi_0, xi_1, xi_2]
    xi = np.array([0, 1/2, 1])

    # custom outer product evaluation function
    eval_outer = lambda f, x: \
        np.frompyfunc(lambda _f, _x: _f(_x), 2, 1).outer(f, x).astype(float)

    return eval_outer(Psis, xi) @ np.diag([1/6, 2/3, 1/6]) @ f(Phi_k(xi, element))


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

    x = FEM_elements.get_nodes(partition)
    hk = FEM_elements.get_element_sizes(partition)

    xi = np.array([0, 1/2, 1])
    fx = f(x)
    fx0 = fx[:-1:2]
    fx1 = fx[1::2]
    fx2 = fx[2::2]

    # Simpsons rule on three points
    F_ks = 1 / 6 * np.array([
        fx0 * Psi[0](xi[0]) + 4 * fx1 * Psi[0](xi[1]) + fx2 * Psi[0](xi[2]),
        fx0 * Psi[1](xi[0]) + 4 * fx1 * Psi[1](xi[1]) + fx2 * Psi[1](xi[2]),
        fx0 * Psi[2](xi[0]) + 4 * fx1 * Psi[2](xi[1]) + fx2 * Psi[2](xi[2]),
    ]).T

    indices = np.array(
        [theta(k, np.array([0, 1, 2])) for k in range(M)]
    )

    np.add.at(F, indices, F_ks * hk[:, None])

    return F

