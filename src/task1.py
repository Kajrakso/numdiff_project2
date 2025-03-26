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

    N = 50
    Ms = np.logspace(1, 4, num=N, dtype=int)

    errs = np.zeros(N)
    for i, M in enumerate(Ms):
        partition = np.linspace(0, 1, M + 1)
        elements = get_elements(partition)
        elements_sizes = get_element_sizes(partition)

        u_h = solve_Poisson_dirichlet(f, a=u(0), b=u(1), partition=partition)
        err = 0
        for k in range(M):

            phi = [
                lambda x: psi[0](physical_element_to_reference_element(x, elements, k)),
                lambda x: psi[1](physical_element_to_reference_element(x, elements, k)),
                lambda x: psi[2](physical_element_to_reference_element(x, elements, k)),
            ]

            a = u_h[local_to_global(k, 0)]
            b = u_h[local_to_global(k, 1)]
            c = u_h[local_to_global(k, 2)]

            x_k = elements[k, 0]
            x_k1 = elements[k, 1]

            integrand = lambda xi: (
                u(reference_element_to_physical_element(xi, elements, k)) -
                (a*psi[0](xi) + b*psi[1](xi) + c*psi[2](xi))
            )**2

            y, _ = sp.integrate.quad(integrand, 0, 1)
            err += np.sqrt(elements_sizes[k]*y)

        errs[i] = err

        # errs[i] = np.sqrt(1/M*np.sum(err**2))

    p = np.polyfit(np.log(1 / Ms), np.log(errs), deg=1)
    eq = (
        r"$"
        + r"\log(E_h) \approx "
        + f"{p[0]:.3f}"
        + r"\log(h)"
        + r"+ \log(C)"
        r"$"
    )

    plt.loglog(1/Ms, errs, 'o-', label=eq)
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

    # plot_solution(M, partition, u_h, u, f_sy)
    convergence_analysis()


if __name__ == "__main__":
    main()


