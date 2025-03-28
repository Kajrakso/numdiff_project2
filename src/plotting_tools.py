import numpy as np
import matplotlib.pyplot as plt

import FEM_elements
from FEM_elements import local_to_global as theta
from FEM_elements import reference_element_to_physical_element as Phi_k
from FEM_elements import Psi



def plot_solution(ax, partition: np.ndarray, u_h: np.ndarray, label: str = None, color="r", K=100) -> None:
    """Plots the approximated solution function `u_h` on the axes `ax`.

    Args:
        ax (Axes): axes to plot on.
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xn`.
        u_h (np.ndarray): solution vector
        label (str): label. Defaults to None.
        color (str, optional): color of the graph. Defaults to 'r'.
        K (int, optional): plotting resolution. Defaults to 100.
    """
    elements = FEM_elements.get_elements(partition)
    
    M = len(elements)
    K_per_elem = K // M
    x_hat = np.linspace(0, 1, K_per_elem)

    x = np.zeros(K_per_elem * M)
    y = np.zeros(K_per_elem * M)
    for k in range(M):
        # extract the coordinates of our basis functions for segment k
        a = u_h[theta(k, 0)]
        b = u_h[theta(k, 1)]
        c = u_h[theta(k, 2)]

        x[K_per_elem * k: K_per_elem * (k+1)] = Phi_k(x_hat, elements[k])
        y[K_per_elem * k: K_per_elem * (k+1)] = a*Psi[0](x_hat) + b*Psi[1](x_hat) + c*Psi[2](x_hat)
    
    ax.plot(x, y, color, label=label)
    ax.plot(partition, u_h[::2], "." + color)


if __name__ == "__main__":
    A = np.array([0, 0.5, 1])

    fig, ax = plt.subplots()
    plot_solution(ax, A, np.array([0, 1, 0, 1, 0]))
    plt.show()
