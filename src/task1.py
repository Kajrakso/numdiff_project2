import scipy as sp
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Any
import FEM_elements
import FEM_poisson
import plotting_tools

from sympy import sin, cos, exp, pi


def manufacture_poisson_functions(
    u_str: str,
) -> tuple[Callable, Any, Callable, Any, Callable, Any]:
    """Manufactures the following functions given a solution (in str format) of the 1D poisson equation: -u_xx = f(x).
    Note: the relevant sympy expressions used have to be imported.
    - u: solution function
    - u_sy: sympy expression for the solution function
    - du: derivative of the solution function
    - du_sy: sympy expression for the derivative of the solution function
    - f: rhs of the 1D Poisson equation: -u_xx = f(x)
    - f_sy: sympy expression for the rhs of the 1D Poisson equation: -u_xx = f(x)

    Args:
        u_str (str): solution function in str format. Make sure the relevant sympy expressions are imported.
            Example: u_str = "sin(x)*exp(-x)"

    Returns:
        tuple[Callable, Any, Callable, Any, Callable, Any]: see description
    """

    print(f"Manufacturing solution for the Poisson equation: -u_xx = f(x)")

    x = sy.symbols("x")

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
    A = FEM_elements.assemble_stiffness_matrix(partition)
    F = FEM_poisson.assemble_load_vector(partition, f)

    FEM_poisson.impose_dirichlet(A, F, a=a, b=b)

    # Solve the equation
    u_h = sp.sparse.linalg.spsolve(A, F)

    # todo: skal vi konstruere en funksjon her, eller bare returnere koordinatene?
    return u_h


def convergence_plot(Ms, errs, filename="conv_plot.pdf"):
    p = np.polyfit(np.log(1 / Ms), np.log(errs), deg=1)

    fig, ax = plt.subplots()
    ax.loglog(1 / Ms, errs, "o-", label=r"$p\approx" + f"{p[0]:.3f}$")
    ax.loglog(1 / Ms, (1 / Ms), label="p=1")
    ax.loglog(1 / Ms, (1 / Ms) ** 2, label="p=2")
    ax.loglog(1 / Ms, (1 / Ms) ** 3, label="p=3")
    ax.legend()
    ax.set(
        title="Convergence plot",
        xlabel="h",
        ylabel=r"$||u-u_h||_{\mathcal{L}^2((0,1))}$",
    )
    fig.savefig(filename)


def convergence_analysis(u, f, Ms):
    N = len(Ms)
    errs = np.zeros(N)
    for i, M in enumerate(Ms):
        partition = np.linspace(0, 1, M + 1)
        elements = FEM_elements.get_elements(partition)
        elements_sizes = FEM_elements.get_element_sizes(partition)

        u_h = solve_Poisson_dirichlet(f, a=u(0), b=u(1), partition=partition)
        err = 0
        for k in range(M):
            a = u_h[FEM_elements.local_to_global(k, 0)]
            b = u_h[FEM_elements.local_to_global(k, 1)]
            c = u_h[FEM_elements.local_to_global(k, 2)]

            integrand = (
                lambda xi: (
                    u(
                        FEM_elements.reference_element_to_physical_element(
                            xi, elements[k]
                        )
                    )
                    - FEM_elements.element_solution_polynomial(a, b, c, xi)
                )
                ** 2
            )

            y, _ = sp.integrate.quad(integrand, 0, 1)
            err += elements_sizes[k] * y

        errs[i] = np.sqrt(err)

    return errs


def test_solution():
    # The number of intervals
    M = 10

    # start and end point.
    xi = 0
    xf = 1

    # A uniform grid
    partition = np.linspace(xi, xf, M + 1)

    # A non-uniform grid
    # partition = np.sqrt(partition)

    # Construct test equation for the Poisson equation: -Δu = f
    u_str = "x*x*x * sin(10*x) + exp(-2*x) * cos(3*pi*x)"
    u, _, _, _, f, f_sy = manufacture_poisson_functions(u_str)

    u_h = solve_Poisson_dirichlet(f, a=u(xi), b=u(xf), partition=partition)

    _, ax = plt.subplots()
    plotting_tools.plot_solution(ax, partition, u_h)
    ax.plot(
        np.linspace(xi, xf, 100),
        u(np.linspace(xi, xf, 100)),
        label="exact",
    )
    ax.set(title=r"$-\Delta u = f, f = " + f"{f_sy}$", xlabel="x", ylabel="u(x)")
    ax.legend()
    plt.show()


def main():
    test_solution()

    u_str = "x*x*x*sin(10*x)+exp(-2*x)*cos(3*pi*x)"  # define the test solution
    u, _, _, _, f, _ = manufacture_poisson_functions(u_str)

    Ms = np.logspace(1, 3, num=20, dtype=int)  # define the number of segments
    errs = convergence_analysis(u, f, Ms)
    convergence_plot(Ms, errs, filename="conv_plot.pdf")


if __name__ == "__main__":
    main()
