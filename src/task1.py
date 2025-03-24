import scipy as sp
import matplotlib.pyplot as plt
from elements import *
from poisson_FEM import *


def solve_Poisson_dirichlet(f, a, b, partition):
    # Assemble matrices
    A = assemble_stiffness_matrix(partition)
    F = assemble_load_vector(partition, f)

    # Solve the equation
    impose_dirichlet(A, F, a=a, b=b)
    A = A.tocsr()

    u_h = sp.sparse.linalg.spsolve(A, F)

    # todo: skal vi konstruere en funksjon her, eller bare returnere koordinatene?
    return u_h


def plot_solution(M, partition, u_h, u, f_sy):
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


def convergence_analysis():
    u_str = "x*x*x*sin(10*x)+exp(-2*x)*cos(3*pi*x)"
    u, _, du, _, f, f_sy = manufacture_solution(u_str)

    N = 10
    Ms = np.logspace(1, 3, num=N, dtype=int)

    errs = np.zeros(N)
    for i, M in enumerate(Ms):
        x_hat = np.linspace(0, 1, int(1000 / M), endpoint=False)

        partition = np.linspace(0, 1, M + 1)
        elements = get_elements(partition)

        u_h = solve_Poisson_dirichlet(f, a=u(0), b=u(1), partition=partition)

        for k in range(M):
            a = u_h[local_to_global(k, 0)]
            b = u_h[local_to_global(k, 1)]
            c = u_h[local_to_global(k, 2)]

            x = reference_element_to_physical_element(x_hat, elements, k)

            solution = element_solution_polynomial(a, b, c, x_hat)
            exact = u(x_hat)

            err = exact - solution

        errs[i] = np.sqrt(1/M*np.sum(err**2))

    plt.loglog(1/Ms, errs, 'o-')
    plt.loglog(1/Ms, (1/Ms), label="h")
    plt.loglog(1/Ms, (1/Ms)**2, label="h^2")
    plt.loglog(1/Ms, (1/Ms)**3, label="h^3")
    plt.legend()
    plt.show()


def main():
    # The number of intervals
    M = 5

    # A uniform grid
    partition = np.linspace(0, 1, M+1)

    # A non-uniform grid
    # partition = np.sqrt(partition)

    # Test solution of the Poisson equation: -Δu = f
    u_str = "x*x*x*sin(10*x)+exp(-2*x)*cos(3*pi*x)"
    u, _, _, _, f, f_sy = manufacture_solution(u_str)

    u_h = solve_Poisson_dirichlet(f, a=u(0), b=u(1), partition=partition)

    plot_solution(M, partition, u_h, u, f_sy)
    # convergence_analysis()


if __name__ == "__main__":
    main()


