import numpy as np
import scipy as sp
import sympy as sy
from sympy import sin, cos, exp, pi
from elements import *


def get_element_load_vector(k, f, elements):
    xi0 = 0
    xi1 = 1 / 2
    xi2 = 1

    x0 = reference_element_to_physical_element(xi0, elements, k)
    x1 = reference_element_to_physical_element(xi1, elements, k)
    x2 = reference_element_to_physical_element(xi2, elements, k)

    return (1 / 6 * np.array(
        [
            (f(x0) * psi[0](xi0) + 4 * f(x1) * psi[0](xi1) + f(x2) * psi[0](xi2)),
            (f(x0) * psi[1](xi0) + 4 * f(x1) * psi[1](xi1) + f(x2) * psi[1](xi2)),
            (f(x0) * psi[2](xi0) + 4 * f(x1) * psi[2](xi1) + f(x2) * psi[2](xi2)),
        ]))


def assemble_load_vector(partition, f):
    M = len(partition) - 1
    N = 2 * M + 1
    F = np.zeros(N)

    elements = get_elements(partition)
    element_sizes = get_element_sizes(partition)

    for k in range(M):
        F_k = get_element_load_vector(k, f, elements)
        F[local_to_global(k, 0):local_to_global(k + 1, 0) + 1] += F_k * element_sizes[k]

    return F


def manufacture_solution(u_str):
    print(f"Manufacturing solution for the Poisson equation: -Δu = f")

    x, y = sy.symbols("x y")

    u_sy = eval(u_str)
    print(f"u = {u_sy}")

    du_sy = sy.simplify(sy.diff(u_sy, x))
    print(f"u' = {du_sy}")

    f_sy = sy.simplify(-sy.diff(u_sy, x, x))
    print(f"f = {f_sy}")

    u = sy.lambdify((x,), u_sy, modules="numpy")
    du = sy.lambdify((x,), du_sy, modules="numpy")
    f = sy.lambdify((x,), f_sy, modules="numpy")
    return u, u_sy, du, du_sy, f, f_sy


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
