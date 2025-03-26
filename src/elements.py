import numpy as np
import scipy as sp


"""
The shape functions defined on the reference element [0, 1]
"""
psi = (
    lambda x: 2 * x**2 - 3 * x + 1,
    lambda x: -4 * x**2 + 4 * x + 0,
    lambda x: 2 * x**2 - 1 * x + 0,
)

"""
Elemental stiffness matrix
"""
B_k = 1 / 3 * np.array([
    [7, -8, 1],
    [-8, 16, -8],
    [1, -8, 7],
])

F_k = 1 / 30 * np.array([
    [4, 2, -1],
    [2, 16, 2],
    [-1, 2, 4],
])


def element_solution_polynomial(a, b, c, x):
    return a*psi[0](x) + b*psi[1](x) + c*psi[2](x)


def get_nodes(partition: np.ndarray) -> np.ndarray:
    nodes = np.zeros(len(partition)*2 - 1)
    nodes[::2] = partition.copy()
    nodes[1::2] = (partition[:-1] + partition[1:])/2
    return nodes


def get_elements(partition: np.ndarray) -> np.ndarray:
    num_intervals = len(partition) - 1
    return np.concatenate((partition[:-1], partition[1:])).reshape((2, num_intervals)).T


def get_element_sizes(partition: np.ndarray):
    return partition[1:] - partition[:-1]


def reference_element_to_physical_element(x, elements, k):
    x_k = elements[k, 0]
    x_k1 = elements[k, 1]

    return (1 - x) * x_k + x * x_k1


def physical_element_to_reference_element(x, elements, k):
    x_k = elements[k, 0]
    x_k1 = elements[k, 1]

    return (x - x_k) / (x_k1 - x_k)


def local_to_global(k, alpha):
    return 2 * k + alpha


def assemble_matrix(partition, element_matrix):
    M = len(partition) - 1
    N = 2 * M + 1

    element_sizes = get_element_sizes(partition)

    # build the diagonals of the sparse matrix
    diag_0 = np.zeros(N)
    diag_1 = np.zeros(N - 1)
    diag_2 = np.zeros(N - 2)

    global_indices = np.array([
        local_to_global(k, np.array([0, 1, 2])) for k in range(M)
    ])

    # accumulated addition for each of the diagonals
    np.add.at(diag_0, global_indices[:, :3], np.diag(element_matrix, k=0) / element_sizes[:, None])
    np.add.at(diag_1, global_indices[:, :2], np.diag(element_matrix, k=1) / element_sizes[:, None])
    np.add.at(diag_2, global_indices[:, :1], np.diag(element_matrix, k=2) / element_sizes[:, None])

    return sp.sparse.diags_array(
        [diag_2, diag_1, diag_0, diag_1, diag_2],
        offsets=[-2, -1, 0, 1, 2]
    ).tolil()
