import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Callable
from elements import F_k, B_k, assemble_matrix, get_nodes


def interpol_x_2(func: Callable, nodes: np.ndarray) -> np.ndarray:
    return func(nodes)


def OCP(func_des: Callable, partition: np.ndarray, alpha: float) -> None:  # ?
    nodes = get_nodes(partition)[1:-1]
    d_arr = interpol_x_2(func_des, nodes)
    B_mat = assemble_matrix(partition, B_k)[1:-1, 1:-1]
    F_mat = assemble_matrix(partition, F_k)[1:-1, 1:-1]

    F_mat_inv = sp.sparse.linalg.inv(F_mat)
    y_sol = sp.sparse.linalg.spsolve(F_mat - B_mat @ F_mat_inv @ B_mat, F_mat @ d_arr)
    u_sol = sp.sparse.linalg.spsolve(F_mat, B_mat @ y_sol)

    # TODO: improve plotting
    plt.plot(nodes, d_arr, label="d")
    plt.plot(nodes, y_sol, label="y sol")
    plt.plot(nodes, u_sol, label="u sol")
    plt.legend()
    plt.show()


def func_1(x):
    return 0.5 * x * (1 - x)


def func_2(x):
    return np.ones_like(x)


def func_3(x):
    res = np.ones_like(x)
    res[0.25 > x] = 0
    res[0.75 < x] = 0
    return res


if __name__ == "__main__":
    partition = np.linspace(0, 1, 100)
    alpha = 1e-3
    OCP(func_2, partition, alpha)
