import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any

import sympy as sy
from sympy import sin, cos, exp, pi

import FEM_elements
from FEM_elements import reference_element_to_physical_element as Phi_k
from FEM_elements import local_to_global as theta
import FEM_poisson
import plotting_tools


def manufacture_poisson_functions(
    u_str: str, logging=False
) -> tuple[Callable, Any, Callable, Any, Callable, Any]:
    """Manufactures the following functions given a solution (in str format)
    of the 1D poisson equation: -u_xx = f(x).
    Note: the relevant sympy expressions used have to be imported.
    - u: solution function
    - u_sy: sympy expression for the solution function
    - du: derivative of the solution function
    - du_sy: sympy expression for the derivative of the solution function
    - f: rhs of the 1D Poisson equation: -u_xx = f(x)
    - f_sy: sympy expression for the rhs of the 1D Poisson equation: -u_xx = f(x)

    Args:
        u_str (str): solution function in str format.
            Make sure the relevant sympy expressions are imported.
            Example: u_str = "sin(x)*exp(-x)"
        logging (Bool): if True, the manufactured functions will get printed. Defaults to False.

    Returns:
        tuple[Callable, Any, Callable, Any, Callable, Any]: see description
    """
    if logging:
        print(f"Manufacturing solution for the Poisson equation: -u_xx = f(x)")

    x = sy.symbols("x")

    u_sy = eval(u_str)
    du_sy = sy.simplify(sy.diff(u_sy, x))
    f_sy = sy.simplify(-sy.diff(u_sy, x, x))

    if logging:
        print(f"u = {u_sy}")
        print(f"u' = {du_sy}")
        print(f"f = {f_sy}")

    u = sy.lambdify((x,), u_sy, modules="numpy")
    du = sy.lambdify((x,), du_sy, modules="numpy")
    f = sy.lambdify((x,), f_sy, modules="numpy")

    return u, u_sy, du, du_sy, f, f_sy


def convergence_plot(Ms, L2errs, H1errs, filename="conv_plot.pdf"):
    pL2 = np.polyfit(np.log(1 / Ms), np.log(L2errs), deg=1)
    pH1 = np.polyfit(np.log(1 / Ms), np.log(H1errs), deg=1)

    fig, ax = plt.subplots()
    ax.loglog(1 / Ms, L2errs, "o-", label=r"$\mathcal{L}^2((0,1)): p\approx" + f"{pL2[0]:.3f}$")
    ax.loglog(1 / Ms, H1errs, "o-", label=r"$\mathcal{H}^1((0,1)): p\approx" + f"{pH1[0]:.3f}$")
    # ax.loglog(1 / Ms, (1 / Ms), label="p=1")
    # ax.loglog(1 / Ms, (1 / Ms) ** 2, label="p=2")
    # ax.loglog(1 / Ms, (1 / Ms) ** 3, label="p=3")
    ax.legend()
    ax.set(
        title="Numerical verification of convergence order",
        xlabel="h",
        ylabel=r"$||u-u_h||$",
    )
    fig.savefig(filename)


def poly_vec(polys, xi, deg):
    return np.array([
        polys[i](xi) for i in range(deg + 1)
    ])


def calculate_errors(u, du, u_h, partition, Psi, dPsi, deg):
    M = len(partition) - 1          # number of elements

    elements = FEM_elements.get_elements(partition)
    elements_sizes = FEM_elements.get_element_sizes(partition)

    L2err_squared = H1err_squared = 0
    for k in range(M):
        elem = elements[k]
        hk = elements_sizes[k]

        # extract the coordinates of our basis functions for segment k
        u_hk = u_h[theta(k, np.arange(deg + 1), deg=deg)]

        def L2integrand(xi):
            return (u(Phi_k(xi, elem)) - u_hk @ poly_vec(Psi, xi, deg)) ** 2

        def H1integrand(xi):
            return (du(Phi_k(xi, elem)) - 1/hk * u_hk @ poly_vec(dPsi, xi, deg)) ** 2

        L2_y, _ = sp.integrate.quad(L2integrand, 0, 1)
        H1_y, _ = sp.integrate.quad(H1integrand, 0, 1)

        L2err_squared += hk * L2_y
        H1err_squared += hk * (H1_y)

    return np.sqrt(L2err_squared), np.sqrt(L2err_squared + H1err_squared)


def convergence_analysis(u, du, f, Ms, deg):
    N = len(Ms)

    Psi = FEM_elements.construct_Psi(deg=deg)
    dPsi = FEM_elements.construct_dPsi(deg=deg)

    L2errs = np.zeros(N)
    H1errs = np.zeros(N)
    for i, M in enumerate(Ms):
        partition = np.linspace(0, 1, M + 1)

        u_h = FEM_poisson.solve_Poisson_dirichlet(f, a=u(0), b=u(1), partition=partition, deg=deg)

        L2_err, H1_err = calculate_errors(u, du, u_h, partition, Psi, dPsi, deg)

        L2errs[i] = L2_err
        H1errs[i] = H1_err

    return Ms, L2errs, H1errs


def test_solution(filename="task1_test_sol.pdf"):
    # degree of the basis polynomials
    deg = 2

    # The number of elements
    M = (4, 6)

    # start and end point.
    xi, xf = 0, 1

    # Construct test equation for the Poisson equation: -Δu = f
    u_str = "x*(x-1)*sin(3*pi*x)"
    u, _, _, _, f, f_sy = manufacture_poisson_functions(u_str)


    fig, axs = plt.subplots(1, len(M), figsize=(10, 6))
    for i, _M in enumerate(M):
        # A uniform grid
        partition = np.linspace(xi, xf, _M + 1)

        # A non-uniform grid
        # partition = np.sqrt(partition)

        u_h = FEM_poisson.solve_Poisson_dirichlet(
            f, a=u(xi), b=u(xf), partition=partition, deg=deg
        )

        axs[i].plot(
            np.linspace(xi, xf, 100),
            u(np.linspace(xi, xf, 100)),
            label="u(x)",
        )
        plotting_tools.plot_solution(axs[i], partition, u_h, label="$u_h(x)$", K=100, deg=deg)
        axs[i].set(
            title=f"$M = {_M}$ segments",
            xlabel="$x$",
            ylabel="$u(x), u_h(x)$",
        )
        axs[i].legend()
    fig.suptitle(r"$-\Delta u = f, u(x) = " + f"{u_str}" + "$")
    fig.savefig(filename)


def main():
    test_solution()

    u_str = "x*(x-1)*sin(3*pi*x)"  # define the test solution
    u, _, du, _, f, _ = manufacture_poisson_functions(u_str)

    Ms = np.logspace(1, 3.5, num=10, dtype=int)  # define the number of segments
    Ms, L2errs, H1errs = convergence_analysis(u, du, f, Ms, deg=2)
    convergence_plot(Ms, L2errs, H1errs, filename="task1_conv_plot.pdf")


if __name__ == "__main__":
    main()

