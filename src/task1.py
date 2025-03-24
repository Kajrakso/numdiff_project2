import numpy as np
import scipy as sp
import sympy as sy
from sympy import sin, cos, exp, pi
import matplotlib.pyplot as plt

"""
The shape functions defined on a reference element [0, 1]
"""
psi = (
    lambda x: 2 * x**2 - 3 * x + 1,
    lambda x: -4 * x**2 + 4 * x + 0,
    lambda x: 2 * x**2 - 1 * x + 0,
)

def element_solution_polynomial(a, b, c, x):
    return a*psi[0](x) + b*psi[1](x) + c*psi[2](x)


def get_elements(nodes: np.ndarray) -> np.ndarray:
    num_intervals = len(nodes) - 1
    return np.concatenate((nodes[:-1], nodes[1:])).reshape((2, num_intervals)).T


def get_element_sizes(nodes: np.ndarray):
    return nodes[1:] - nodes[:-1]


def reference_element_to_physical_element(x, elements, k):
    x_k = elements[k, 0]
    x_k1 = elements[k, 1]

    return (1 - x) * x_k + x * x_k1


def local_to_global(k, alpha):
    return 2 * k + alpha


def get_element_load_vector(k, f, elements):
    xi0 = 0
    xi1 = 1 / 2
    xi2 = 1

    x0 = reference_element_to_physical_element(xi0, elements, k)
    x1 = reference_element_to_physical_element(xi1, elements, k)
    x2 = reference_element_to_physical_element(xi2, elements, k)

    return (1 / 6 * np.array(
        [
            (f(x0) * psi[0](xi0) + 4 * f(x1) * psi[0](xi1) + f(x2) * psi[0](xi2)),
            (f(x0) * psi[1](xi0) + 4 * f(x1) * psi[1](xi1) + f(x2) * psi[1](xi2)),
            (f(x0) * psi[2](xi0) + 4 * f(x1) * psi[2](xi1) + f(x2) * psi[2](xi2)),
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

    return sp.sparse.diags_array(
        [diag_2, diag_1, diag_0, diag_1, diag_2],
        offsets=[-2, -1, 0, 1, 2]
    ).tolil()


def assemble_load_vector(partition, f):
    M = len(partition) - 1
    N = 2 * M + 1
    F = np.zeros(N)

    elements = get_elements(partition)
    element_sizes = get_element_sizes(partition)

    for k in range(M):
        F_k = get_element_load_vector(k, f, elements)
        F[local_to_global(k, 0):local_to_global(k + 1, 0) + 1] += F_k * element_sizes[k]

    return F


def manufacture_solution(u_str):
    print(f"Manufacturing solution for the Poisson equation: -Δu = f")

    x, y = sy.symbols("x y")

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


def impose_dirichlet(A, F, a, b):
    """u(0) = a, u(1) = b"""
    A[0, :] = 0
    A[-1, :] = 0
    A[0, 0] = 1
    A[-1, -1] = 1

    F[0] = a
    F[-1] = b


def impose_neumann_dirichlet(A, F, a, b):
    """u'(0) = a, u(1) = b"""
    F[0] -= a

    A[-1, :] = 0
    A[-1, -1] = 1

    F[-1] = b


def impose_dirichlet_neumann(A, F, a, b):
    """u(0) = a, u'(1) = b"""
    F[-1] += b

    A[0, :] = 0
    A[0, 0] = 1

    F[0] = a


def main():
    # The number of intervals
    M = 6

    # A uniform grid
    partition = np.linspace(0, 1, M+1)

    # A non-uniform grid
    partition = np.sqrt(partition)


    # Test solution of the Poisson equation: -Δu = f
    u_str = "exp(x) + cos(pi*x)/exp(x)"
    u, _, du, _, f, f_sy = manufacture_solution(u_str)

    # Assemble matrices
    A = assemble_stiffness_matrix(partition)
    F = assemble_load_vector(partition, f)

    # Solve the equation
    impose_dirichlet(A, F, a=u(0), b=u(1))
    A = A.tocsr()
    u_h = sp.sparse.linalg.spsolve(A, F)

    # plot the solution
    x_hat = np.linspace(0, 1, 10)
    elements = get_elements(partition)

    fig, ax = plt.subplots()

    for k in range(M):
        a = u_h[local_to_global(k, 0)]
        b = u_h[local_to_global(k, 1)]
        c = u_h[local_to_global(k, 2)]

        x = reference_element_to_physical_element(x_hat, elements, k)

        ax.plot(x, element_solution_polynomial(a, b, c, x_hat))
        ax.plot(x[-1], element_solution_polynomial(a, b, c, x_hat[-1]), 'o')


    ax.plot(
        np.linspace(0, 1, 100),
        u(np.linspace(0, 1, 100)),
        label="exact",
    )
    ax.set(
        title=r"$-\Delta u = f, f = " + f"{f_sy}$",
        xlabel="x",
        ylabel="u(x)"
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()


