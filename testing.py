from step_selection import backtracking, wolfe
from test_functions import rosenbrock, himmelblau, ackley, rastrigin
from descent import Descent
import numpy as np
import matplotlib.pyplot as plt


from modification_methods import *

def rosenbrock_testing():
    x0 = np.array([0.5, 0.5])
    # x0 = np.array([1.2, 1.2]).T

    r = Descent(rosenbrock)

    # sol = r.descend(x0, r.steepest, backtracking, display=True)
    sol = r.descend(x0, r.newton, backtracking, display=True)
    # sol = r.descend(x0, r.steepest, wolfe, display=True)

    print(sol)


def himmelblau_testing():
    """
    Newton method doesn't work, I suspect because hessian is not positive definite
    """
    h = Descent(himmelblau)

    p1 = np.array([0.5, 0.5])
    p2 = np.array([-5., 5.])
    p3 = np.array([-5., -5.])
    p4 = np.array([3., -2.])

    p = np.array([-0.3, -3.2])

    eq = h.descend(p, h.steepest, backtracking)  # converges to 1st equilibrium
    print(eq)


    eq1 = h.descend(p1, h.steepest, backtracking)  # converges to 1st equilibrium
    eq2 = h.descend(p2, h.steepest, backtracking)  # converges to 2rd equilibrium
    eq3 = h.descend(p3, h.steepest, backtracking)  # converges to 3nd equilibrium
    eq4 = h.descend(p4, h.steepest, backtracking)  # converges to 4th equilibrium
    print(eq1)
    print(eq2)
    print(eq3)
    print(eq4)

    # eq1 = h.descend(np.array([0.5, 0.5]), h.newton, backtracking, display=True)
    # hess =  himmelblau.hessian(np.array([0.5, 0.5]))
    # print(hess)
    # print(is_positive_definite(hess))

    # Strangely enough, on (-5, -5) wolfe and backtracking converges to different equilibria
    # sol = h.BFGS(p3, step_selection_mode=wolfe, display=True)
    
    # print(sol)


def rastrigin_testing():
    r = Descent(rastrigin)

    x0 = np.array([-0.1, 0.5])
    eq = r.descend(x0, r.newton, wolfe)
    eq2 = r.BFGS(x0)

    print(eq)
    print(eq2)



def himmelblau_plot():
    """
    Displays which equilibrium point each point converges to in a colour plot
    Takes 10 seconds to run for step=0.1
    """

    h = Descent(himmelblau)

    known_eqs = [
        np.array([3.0, 2.0]),
        np.array([-2.805118, 3.131312]),
        np.array([-3.779310, -3.283186]),
        np.array([3.584428, -1.848126])
    ]

    step = 0.1
    minimum = -5.0
    maximum = 5.0

    xs = np.arange(minimum, maximum + step, step)
    ys = np.arange(minimum, maximum + step, step)

    X, Y = np.meshgrid(xs, ys)

    Z = [[0 for _ in range(len(X))] for _ in range(len(Y))]

    for i in range(len(X)):
        for j in range(len(Y)):
            x, y = xs[i], ys[j]
            try:
                Z[i][j] = h.descend2D(x, y, h.steepest, backtracking)
                print(i, j, Z[i][j])
            except:
                Z[i][j] = -1


    sols = get_solution_numbers(known_eqs, Z)

    plt.pcolormesh(X, Y, sols)
    plt.colorbar()
    plt.show()


def get_solution_numbers(known_eqs, Z, tolerance=1e-6):
    def get_solution_number(z):
        for i in range(len(known_eqs)):
            if np.linalg.norm(known_eqs[i] - z) < tolerance:
                return i
        return -1
    
    return [[get_solution_number(Z[i][j]) for i in range(len(Z))] for j in range(len(Z[0]))]



himmelblau_plot()