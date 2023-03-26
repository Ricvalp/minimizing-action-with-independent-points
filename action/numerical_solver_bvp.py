import numpy as np
import scipy
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


def pendulum_bvp(y_0, y_1, T, N):

    def fun(x, y):
        return np.vstack((y[1], -10*np.sin(y[0])))

    def bc(ya, yb):
        return np.array([ya[0] - y_0, yb[0] - y_1])

    x = np.linspace(0, T, N)

    y_a = np.ones((2, x.size))
    y_b = np.ones((2, x.size))
    y_b[0] = 3

    res_a = solve_bvp(fun, bc, x, y_a, max_nodes=1000, tol=0.0001)
    res_b = solve_bvp(fun, bc, x, y_b, max_nodes=1000, tol=0.0001)

    return (x, res_a.sol(x)[0]), (x, res_b.sol(x)[0])


if __name__=="__main__":

    (x, y_plot_a), (x, y_plot_b) = pendulum_bvp(0, -3, 1.5, 100)

    plt.plot(x, y_plot_a, label='y_a')
    plt.plot(x, y_plot_b, label='y_b')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()