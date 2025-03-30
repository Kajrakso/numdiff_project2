import numpy as np
import matplotlib.pyplot as plt

import FEM_elements
from FEM_elements import local_to_global as theta
from FEM_elements import reference_element_to_physical_element as Phi_k


def plot_solution(ax, partition: np.ndarray, u_h: np.ndarray, deg: int, label: str = None, color="r", K=100) -> None:
    M = len(partition) - 1
    K_per_elem = K // M                         # number of points for plotting per segment
    x_hat = np.linspace(0, 1, K_per_elem)       # reference element used for plotting

    elements = FEM_elements.get_elements(partition)
    Psi = FEM_elements.construct_Psi(deg=deg)

    poly_vec = np.array([
        Psi[i](x_hat) for i in range(deg + 1)
    ])

    # prepare arrays for plotting
    x = np.zeros(K_per_elem * M)
    y = np.zeros(K_per_elem * M)

    for k in range(M):
        # extract the coordinates of our basis functions for segment k
        coords = u_h[theta(k, np.arange(deg + 1), deg=deg)]


        # transform the reference x-element to the physical element
        x[K_per_elem * k: K_per_elem * (k+1)] = Phi_k(x_hat, elements[k])

        # sum up those polynomials!
        y[K_per_elem * k: K_per_elem * (k+1)] = np.sum(coords[:, None]*poly_vec, axis=0)

    ax.plot(x, y, color, label=label)
    ax.plot(partition, u_h[::deg], "." + color)


if __name__ == "__main__":
    A = np.array([0, 0.5, 1])

    fig, ax = plt.subplots()
    plot_solution(ax, A, np.array([0, 1, -1, 0, 1, 1, 2]), K=100, deg=3)
    plt.show()
