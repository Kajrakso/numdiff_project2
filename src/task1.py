import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def get_elements(nodes: np.ndarray) -> np.ndarray:
    num_intervals = len(nodes) - 1
    return np.concatenate((nodes[:-1], nodes[1:])).reshape((2, num_intervals)).T


def get_element_sizes(nodes: np.ndarray) -> np.ndarray:
    return nodes[1:] - nodes[:-1]


def reference_element_to_physical_element(x, elements, k):
    x_k = elements[k, 0]
    x_k1 = elements[k, 1]

    return (1 - x) * x_k + x * x_k1


def local_to_global(k, alpha):
    return 2 * k + alpha


def get_element_load_vector(k, f, phi, elements):
    xi0 = 0
    xi1 = 1 / 2
    xi2 = 1

    x0 = reference_element_to_physical_element(xi0, elements, k)
    x1 = reference_element_to_physical_element(xi1, elements, k)
    x2 = reference_element_to_physical_element(xi2, elements, k)

    return (1 / 6 * np.array(
        [
            (f(x0) * phi[0](xi0) + 4 * f(x1) * phi[0](xi1) + f(x2) * phi[0](xi2)),
            (f(x0) * phi[1](xi0) + 4 * f(x1) * phi[1](xi1) + f(x2) * phi[1](xi2)),
            (f(x0) * phi[2](xi0) + 4 * f(x1) * phi[2](xi1) + f(x2) * phi[2](xi2)),
        ]))


def assemble_stiffness_matrix(partition):
    M = len(partition) - 1
    N = 2 * M + 1
    B = np.zeros((N, N))

    element_sizes = get_element_sizes(partition)

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

    global_indices = np.array([
        local_to_global(k, np.array([0, 1, 2])) for k in range(M)
    ])

    # accumulated addition for each of the diagonals
    np.add.at(diag_0, global_indices[:, :3], np.diag(B_k, k=0) / element_sizes[:, None])
    np.add.at(diag_1, global_indices[:, :2], np.diag(B_k, k=1) / element_sizes[:, None])
    np.add.at(diag_2, global_indices[:, :1], np.diag(B_k, k=2) / element_sizes[:, None])

    # todo: proper boundary condition handeling
    diag_2 = diag_2[1:-1]
    diag_1 = diag_1[1:-1]
    diag_0 = diag_0[1:-1]

    return sp.sparse.diags_array(
        [diag_2, diag_1, diag_0, diag_1, diag_2],
        offsets=[-2, -1, 0, 1, 2]
    ).tocsr()


def assemble_load_vector(partition, f):
    M = len(partition) - 1
    N = 2 * M + 1
    F = np.zeros(N)

    elements = get_elements(partition)
    element_sizes = get_element_sizes(partition)

    phi = (
        lambda x: 2 * x**2 - 3 * x + 1,
        lambda x: -4 * x**2 + 4 * x + 0,
        lambda x: 2 * x**2 - 1 * x + 0,
    )

    for k in range(M):
        F_k = get_element_load_vector(k, f, phi, elements)
        F[2 * k:2 * (k + 1) + 1] += F_k * element_sizes[k]

    return F


def main():
    M = 4  # the number of intervals

    # partition = np.linspace(0, 1, M+1)              # a uniform grid
    partition = np.array([0, 0.5, 0.6, 0.9, 1])     # a non uniform grid

    # The RHS of the Poisson equation: -Δu = f
    def f(x):
        return np.ones_like(x)

    F = assemble_stiffness_matrix(partition)
    B = assemble_load_vector(partition, f)

    # temp: only support for dirichlet boundaries:
    # impose boundary contitions by just ignoring them...
    # u(0) = u(1) = 0
    solution = np.zeros_like(B)
    solution[1:-1] = sp.sparse.linalg.spsolve(F, B[1:-1])

    # just plotting
    fig, ax = plt.subplots()
    ax.plot(
        np.linspace(0, 1, 100),
        (lambda x: -1 / 2 * (x - 1) * x)(np.linspace(0, 1, 100)),
        label="exact",
    )
    ax.plot(partition, solution[::2], "-o", label="FEM")
    ax.legend()
    ax.set(title=r"$-\Delta u = f, f = 1$", xlabel="x", ylabel="u(x)")
    plt.show()


if __name__ == "__main__":
    main()


