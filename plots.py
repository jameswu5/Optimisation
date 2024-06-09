import numpy as np
import matplotlib.pyplot as plt

from test_functions import rosenbrock, himmelblau, ackley, rastrigin


def rosenbrock_2D(x, y):
    return rosenbrock.func(np.array([x, y]))

def himmelblau_2D(x, y):
    return himmelblau.func(np.array([x, y]))

def ackley_2D(x, y):
    return ackley.func(np.array([x, y]))

def rastrigin_2D(x, y):
    return rastrigin.func(np.array([x, y]))

def contour_plot(f, b, levels):
    """
    b (int): min and max values; the boundaries
    f (func): function from R^2 to R
    """
    x = np.linspace(-b, b, 100)
    y = np.linspace(-b, b, 100)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    plt.contour(X, Y, Z, levels=levels)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()
    # plt.savefig(f"images/{f.__name__}.png")

# contour_plot(rosenbrock_2D, 5, range(0, 10, 1))
# contour_plot(himmelblau_2D, 5, range(0, 101, 10))
# contour_plot(rastrigin_2D, 5, range(0, 41, 5))
# contour_plot(ackley_2D, 5, range(10))
