import numpy as np
from numpy import polynomial as poly


def construct_Psi(deg: int) -> np.ndarray[poly.Polynomial]:
    """The shape functions of degree deg
    defined on the reference element [0, 1] """
    nodes = np.linspace(0, 1, deg + 1)

    p = poly.polynomial.polyvander(nodes, deg)
    A = np.linalg.inv(p)

    return [poly.Polynomial(_A) for _A in A.T]


def construct_dPsi(deg: int) -> np.ndarray[poly.Polynomial]:
    """The derivative of the shape functions of
    degree deg defined on the reference element [0, 1]."""
    nodes = np.linspace(0, 1, deg + 1)

    p = poly.polynomial.polyvander(nodes, deg)
    A = np.linalg.inv(p)

    return [poly.Polynomial(_A).deriv(1) for _A in A.T]


def get_nodes(partition: np.ndarray, deg: int) -> np.ndarray:
    """Given a partition x0 < x1 < ... < xM,
    construct a array of nodes by splitting each element into deg equally sized elements.
    Example (deg=2): `[0, 0.6, 1] => [0, 0.3, 0.6, 0.8, 1]`.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.
        deg (int): degree n gives n + 1 nodes per element

    Returns:
        np.ndarray: array of nodes
    """
    M = len(partition) - 1
    N = M*deg + 1

    nodes = np.zeros(N)
    nodes[-1] = partition[-1]       # add the endpoint
    nodes[:-1] = np.linspace(partition[:-1], partition[1:], deg, endpoint=False, axis=1).flatten()
    return nodes


def get_elements(partition: np.ndarray) -> np.ndarray:
    """Construct array of elements given a partition.
    Example: `[0, 0.5, 1] => [[0, 0.5], [0.5, 1]]`.

    Args:
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xM`.

    Returns:
        np.ndarray: elements on the interval.
    """
    N = len(partition) - 1
    return np.concatenate((partition[:-1], partition[1:])).reshape((2, N)).T


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
    if element.ndim > 1:
        x_k = element[..., 0][:, None]
        x_k1 = element[..., 1][:, None]
    else:
        x_k = element[0]
        x_k1 = element[1]

    res = (1 - xi) * x_k + xi * x_k1
    return res


def physical_element_to_reference_element(
    x: float | np.ndarray, element: np.ndarray
) -> np.ndarray:
    if element.ndim > 1:
        x_k = element[..., 0][:, None]
        x_k1 = element[..., 1][:, None]
    else:
        x_k = element[0]
        x_k1 = element[1]

    return (x - x_k) / (x_k1 - x_k)


def local_to_global(k: int, alpha: int, deg: int) -> int:
    r"""For the k'th segments and the alpha'th basis function on the interval,
    return the global index.

    Args:
        k (int): segment
        alpha (int): local basis function: alpha \in \{0, 1, ..., deg\}.
        deg (int): polynomial degree of the basis functions.

    Returns:
        int: global index
    """
    return deg * k + alpha


if __name__ == "__main__":
    xi = np.array([0.0, 0.5, 1.0])
    element = np.array([0, 1])
    element = np.array([[0, 1], [1, 2]])

    print(reference_element_to_physical_element(xi , element))

