import numpy as np
import scipy as sp


"""The shape functions defined on the reference element [0, 1] """
psi = (
    lambda x: 2 * x**2 - 3 * x + 1,
    lambda x: -4 * x**2 + 4 * x + 0,
    lambda x: 2 * x**2 - 1 * x + 0,
)


def element_solution_polynomial(
    a: float, b: float, c: float, x: float | np.ndarray
) -> float | np.ndarray:
    """Evaluate the Lagrange interpolating polynomial at x
    defined by the points (0, a), (1/2, b), (1, c).

    Args:
        a (float): y coord of the point (0, a)
        b (float): y coord of the point (1/2, b)
        c (float): y coord of the point (1, c)
        x (float | np.ndarray): x

    Returns:
        float | np.ndarray: y
    """
    return a * psi[0](x) + b * psi[1](x) + c * psi[2](x)


def get_nodes(partition: np.ndarray) -> np.ndarray:
    """Given a partition x0 < x1 < ... < xM,
    construct a array of nodes by adding the middle
    point in each element. Example: `[0, 0.6, 1] => [0, 0.3, 0.6, 0.8, 1]`.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.

    Returns:
        np.ndarray: array of nodes
    """
    nodes = np.zeros(len(partition) * 2 - 1)
    nodes[::2] = partition.copy()
    nodes[1::2] = (partition[:-1] + partition[1:]) / 2
    return nodes


def get_elements(partition: np.ndarray) -> np.ndarray:
    """Construct array of elements given a partition.
    Example: `[0, 0.5, 1] => [[0, 0.5], [0.5, 1]]`.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.

    Returns:
        np.ndarray: elements on the interval.
    """
    num_intervals = len(partition) - 1
    return np.concatenate((partition[:-1], partition[1:])).reshape((2, num_intervals)).T


def get_element_sizes(partition: np.ndarray) -> np.ndarray:
    """Construct an array of element sizes
    given a partition of an interval.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.

    Returns:
        np.ndarray: element sizes
    """
    return partition[1:] - partition[:-1]


def reference_element_to_physical_element(
    xi: float | np.ndarray, element: np.ndarray
) -> np.ndarray:
    x_k = element[0]
    x_k1 = element[1]

    return (1 - xi) * x_k + xi * x_k1


def physical_element_to_reference_element(
    x: float | np.ndarray, element: np.ndarray
) -> np.ndarray:
    x_k = element[0]
    x_k1 = element[1]

    return (x - x_k) / (x_k1 - x_k)


def local_to_global(k: int, alpha: int) -> int:
    r"""For the k'th segments and the alpha'th basis function on the interval,
    return the global index.

    Args:
        k (int): segment
        alpha (int): local basis function: alpha \in \{0, 1, 2\}.

    Returns:
        int: global index
    """
    return 2 * k + alpha


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

    element_sizes = get_element_sizes(partition)

    # elemental mass matrix
    F_k = (
        1
        / 30
        * np.array(
            [
                [4, 2, -1],
                [2, 16, 2],
                [-1, 2, 4],
            ]
        )
    )

    # build the diagonals of the sparse matrix
    diag_0 = np.zeros(N)
    diag_1 = np.zeros(N - 1)
    diag_2 = np.zeros(N - 2)

    global_indices = np.array(
        [local_to_global(k, np.array([0, 1, 2])) for k in range(M)]
    )

    if remove_boundary:
        diag_0 = diag_0[1:-1]
        diag_1 = diag_1[1:-1]
        diag_2 = diag_2[1:-1]

    # accumulated addition for each of the diagonals
    np.add.at(diag_0, global_indices[:, :3], np.diag(F_k, k=0) * element_sizes[:, None])
    np.add.at(diag_1, global_indices[:, :2], np.diag(F_k, k=1) * element_sizes[:, None])
    np.add.at(diag_2, global_indices[:, :1], np.diag(F_k, k=2) * element_sizes[:, None])

    return sp.sparse.diags_array(
        [diag_2, diag_1, diag_0, diag_1, diag_2], offsets=[-2, -1, 0, 1, 2]
    ).tocsc()


def assemble_stiffness_matrix(
    partition: np.ndarray, remove_boundary=False
) -> np.ndarray:
    """Assembles the stiffness matrix.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        remove_boundary (bool, optional): If True, the first and last rows and columns are removes from the matrix. Defaults to False.

    Returns:
        np.ndarray: stiffness matrix
    """
    M = len(partition) - 1
    N = 2 * M + 1

    element_sizes = get_element_sizes(partition)

    # elemental stiffness matrix
    B_k = (
        1
        / 3
        * np.array(
            [
                [7, -8, 1],
                [-8, 16, -8],
                [1, -8, 7],
            ]
        )
    )

    # build the diagonals of the sparse matrix
    diag_0 = np.zeros(N)
    diag_1 = np.zeros(N - 1)
    diag_2 = np.zeros(N - 2)

    global_indices = np.array(
        [local_to_global(k, np.array([0, 1, 2])) for k in range(M)]
    )

    # accumulated addition for each of the diagonals
    np.add.at(diag_0, global_indices[:, :3], np.diag(B_k, k=0) / element_sizes[:, None])
    np.add.at(diag_1, global_indices[:, :2], np.diag(B_k, k=1) / element_sizes[:, None])
    np.add.at(diag_2, global_indices[:, :1], np.diag(B_k, k=2) / element_sizes[:, None])

    if remove_boundary:
        diag_0 = diag_0[1:-1]
        diag_1 = diag_1[1:-1]
        diag_2 = diag_2[1:-1]

    return sp.sparse.diags_array(
        [diag_2, diag_1, diag_0, diag_1, diag_2], offsets=[-2, -1, 0, 1, 2]
    ).tocsc()


if __name__ == "__main__":
    part = np.array([0, 0.6, 1])
    print(get_nodes(part))
