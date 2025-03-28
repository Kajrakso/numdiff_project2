import numpy as np
import scipy as sp
from typing import Callable

"""The shape functions defined on the reference element [0, 1] """
Psi = (
    lambda x: 2 * x**2 - 3 * x + 1,
    lambda x: -4 * x**2 + 4 * x + 0,
    lambda x: 2 * x**2 - 1 * x + 0,
)

"""The derivative of the shape functions defined on the reference element [0, 1] """
dPsi = (
    lambda x: 4 * x - 3,
    lambda x: -8 * x + 4,
    lambda x: 4 * x - 1,
)


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


if __name__ == "__main__":
    pass