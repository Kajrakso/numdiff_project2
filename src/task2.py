import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Callable
from FEM_elements import get_nodes
from FEM_assemble import assemble_stiffness_matrix, assemble_mass_matrix
import plotting_tools


def OCP(func_des: Callable, partition: np.ndarray, alpha: float, deg: int) -> np.ndarray:
    inner_nodes = get_nodes(partition, deg=deg)[1:-1]
    N = len(inner_nodes)

    B = assemble_stiffness_matrix(partition, deg=deg)[1:-1, 1:-1].tocsr()
    F = assemble_mass_matrix(partition, deg=deg)[1:-1, 1:-1].tocsr()

    A = sp.sparse.block_array([[F, alpha * B], [-B, F]])
    b = np.concatenate([F @ func_des(inner_nodes), np.zeros(N)])

    sol = sp.sparse.linalg.spsolve(A, b)

    y = np.concatenate([[0], sol[:N], [0]])
    u = np.concatenate([[0], sol[N:], [0]])

    return y, u


def task2() -> None:
    def yd1(x):
        r"""$y_{d}(x) = \frac{1}{2}x(1-x)$"""
        return 0.5 * x * (1 - x)

    def yd2(x):
        r"""$y_d(x) = 1$"""
        return np.ones_like(x)

    def yd3(x):
        r"""$y_d(x) = 1, \{1/4 < x < 3/4\}$"""
        res = np.zeros_like(x)
        res[(1 / 4 <= x) & (x <= 3 / 4)] = 1
        return res

    # degree of the basis polynomials
    deg = 2

    # The number of elements
    M = 20

    # start and end point
    xi, xf = 0, 1

    # different alpha values
    alphas = (1e-1, 1e-3, 1e-8)

    partition = np.linspace(xi, xf, M + 1)
    x = np.linspace(xi, xf, 100)

    for k, alpha in enumerate(alphas):
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        for i, yd in enumerate((yd1, yd2, yd3)):
            y_sol, u_sol = OCP(yd, partition, alpha, deg=deg)

            axs[0, i].plot(x, yd(x), label="desired temp. profile")
            plotting_tools.plot_solution(axs[0, i], partition, y_sol, deg=deg, color="r")
            axs[0, i].set(xlim=(xi, xf), title=yd.__doc__)

            plotting_tools.plot_solution(axs[1, i], partition, u_sol, deg=deg, color="r")
            axs[1, i].set(xlim=(xi, xf), title="Distributed heat source $u$")

            axs[0, i].legend()

        fig.suptitle(
            r"OCP: partitioning of "
            + f"$[{xi}, {xf}]$ into ${M}$"
            + r" segments, cost parameter: $\alpha=$"
            + f"{alpha:.0e}. deg = {deg}."
        )
        fig.tight_layout()
        fig.savefig(f"task2_fig_{k}.pdf")


if __name__ == "__main__":
    task2()
