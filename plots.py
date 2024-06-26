import numpy as np
import matplotlib.pyplot as plt

from test_functions import rosenbrock, himmelblau, ackley, rastrigin, sphere


def rosenbrock_2D(x, y):
    return rosenbrock.func(np.array([x, y]))

def himmelblau_2D(x, y):
    return himmelblau.func(np.array([x, y]))

def ackley_2D(x, y):
    return ackley.func(np.array([x, y]))

def rastrigin_2D(x, y):
    return rastrigin.func(np.array([x, y]))

def sphere_2D(x, y):
    return sphere.func(np.array([x, y]))

def contour_plot(f, xmin, xmax, ymin, ymax, levels):
    """
    b (int): min and max values; the boundaries
    f (func): function from R^2 to R
    """
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x, y)

    Z = f(X, Y)

    plt.contour(X, Y, Z, levels=levels)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar()
    plt.show()
    # plt.savefig(f"images/{f.__name__}_2.png")

# contour_plot(rosenbrock_2D, -2, 2, -1, 3, range(0, 10, 1))
# contour_plot(himmelblau_2D, -5, 5, -5, 5, range(0, 101, 10))
# contour_plot(rastrigin_2D, -5, 5, -5, 5, range(0, 41, 5))
# contour_plot(ackley_2D, -5, 5, -5, 5, range(10))
contour_plot(sphere_2D, -4, 4, -4, 4, range(0, 21, 3))