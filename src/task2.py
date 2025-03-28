import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Callable
from FEM_elements import assemble_stiffness_matrix, assemble_mass_matrix, get_nodes
import plotting_tools


def OCP(func_des: Callable, partition: np.ndarray, alpha: float) -> np.ndarray:
    nodes = get_nodes(partition)
    N = len(nodes) - 2

    B = assemble_stiffness_matrix(partition, remove_boundary=True)
    F_hat = assemble_mass_matrix(partition)[1:-1, :]
    F = F_hat[:, 1:-1]

    A = sp.sparse.block_array([[F, alpha * B], [-B, F]])

    d = func_des(nodes)
    b = np.concatenate([F_hat @ d, np.zeros(N)])

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

    M = 10
    alpha = 1e-3

    partition = np.linspace(0, 1, M + 1)
    nodes = get_nodes(partition)
    x = np.linspace(0, 1, 100)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for i, yd in enumerate((yd1, yd2, yd3)):
        y_sol, u_sol = OCP(yd, partition, alpha)

        axs[0, i].plot(x, yd(x), label="desired temp. profile")

        plotting_tools.plot_solution(axs[0, i], partition, y_sol)

        axs[0, i].set(xlim=(0, 1), title=yd.__doc__)

        axs[1, i].plot(nodes, u_sol, "r")
        axs[1, i].set(xlim=(0, 1), title="Distributed heat source $u$")

        axs[0, i].legend()

    fig.suptitle(
        r"OCP: partitioning of "
        + f"$[0, 1]$ into ${M}$"
        + r" segments, cost parameter: $\alpha=$"
        + f"{alpha:.0e}."
    )
    fig.tight_layout()
    fig.savefig(f"fig{alpha}.pdf")


if __name__ == "__main__":
    task2()
