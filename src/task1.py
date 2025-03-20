import numpy as np
import matplotlib.pyplot as plt


def get_elements(nodes: np.ndarray) -> np.ndarray:
    num_intervals = len(nodes) - 1
    return np.concatenate((nodes[:-1], nodes[1:])).reshape((2, num_intervals)).T


def get_element_sizes(partition: np.ndarray) -> np.ndarray:
    return partition[:, 1] - partition[:, 0]


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

    elements = get_elements(partition)
    element_sizes = get_element_sizes(elements)

    # elemental stiffness matrix
    B_k = (1 / 3) * np.array([
        [7, -8, 1],
        [-8, 16, -8],
        [1, -8, 7],
    ])

    for k in range(M):
        i0 = local_to_global(k, 0)
        i2 = local_to_global(k, 2)

        B[i0:i2 + 1, i0:i2 + 1] += B_k / element_sizes[k]

    return B


def assemble_load_vector(partition, f):
    M = len(partition) - 1
    N = 2 * M + 1
    F = np.zeros(N)

    elements = get_elements(partition)
    element_sizes = get_element_sizes(elements)

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

    partition = np.linspace(0, 1, M + 1)

    # The RHS of the Poisson equation: -Δu = f
    def f(x):
        return np.ones_like(x)

    F = assemble_stiffness_matrix(partition)
    B = assemble_load_vector(partition, f)

    # temp: only support for dirichlet boundaries:
    # impose boundary contitions by just ignoring them...
    # u(0) = u(1) = 0
    solution = np.zeros_like(B)
    solution[1:-1] = np.linalg.solve(F[1:-1, 1:-1], B[1:-1])

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
