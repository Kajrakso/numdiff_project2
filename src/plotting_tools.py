import numpy as np
import matplotlib.pyplot as plt
import FEM_elements


def plot_solution(ax, partition: np.ndarray, u_h: np.ndarray, color="r", K=100) -> None:
    """Plots the approximated solution function `u_h` on the axes `ax`.

    Args:
        ax (Axes): axes to plot on.
        partition (np.ndarray): partition of an interval: `x0 < x1 < ... < xn`.
        u_h (np.ndarray): solution vector
        color (str, optional): color of the graph. Defaults to 'r'.
        K (int, optional): plotting resolution. Defaults to 100.
    """
    elements = FEM_elements.get_elements(partition)
    x_hat = np.linspace(0, 1, int(K / len(elements)))

    for k in range(len(partition) - 1):
        # extract the coordinates of our basis functions for segment k
        a = u_h[FEM_elements.local_to_global(k, 0)]
        b = u_h[FEM_elements.local_to_global(k, 1)]
        c = u_h[FEM_elements.local_to_global(k, 2)]

        x = FEM_elements.reference_element_to_physical_element(x_hat, elements[k])

        ax.plot(x, FEM_elements.element_solution_polynomial(a, b, c, x_hat), color)

    ax.plot(partition, u_h[::2], "o" + color)


if __name__ == "__main__":
    A = np.array([0, 0.5, 1])

    fig, ax = plt.subplots()
    plot_solution(ax, A, np.array([0, 1, 0, 1, 0]))
    plt.show()
