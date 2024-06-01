from step_selection import backtracking, wolfe
from test_functions import rosenbrock, himmelblau, ackley, rastrigin
from descent import Descent
import numpy as np


def main():
    x0 = np.array([0.5, 0.5])
    # x0 = np.array([1.2, 1.2]).T

    r = Descent(rosenbrock)
    a = Descent(ackley)

    # sol = r.descend(x0, r.steepest, backtracking, display=True)
    # sol = r.descend(x0, r.newton, backtracking, display=True)
    # sol = r.descend(x0, r.steepest, wolfe, display=True)


def himmelblau_testing():
    """
    As of right now:

    Works on steepest descent with backtracking and wolfe
    Doesn't work on Newton at all
    Works on BFGS on only backtracking

    TODO: check the hessian
    """
    # Himmelblau equilibria
    h = Descent(himmelblau)

    p1 = np.array([0.5, 0.5])
    p2 = np.array([-5., 5.])
    p3 = np.array([-5., -5.])
    p4 = np.array([3., -2.])

    eq1 = h.descend(p1, h.steepest, backtracking)  # converges to 1st equilibrium
    eq2 = h.descend(p2, h.steepest, backtracking)  # converges to 3rd equilibrium
    eq3 = h.descend(p3, h.steepest, backtracking)  # converges to 2nd equilibrium
    eq4 = h.descend(p4, h.steepest, backtracking)  # converges to 3rd equilibrium

    print(eq1)
    print(eq2)
    print(eq3)
    print(eq4)

    # sol = h.BFGS(p3, step_selection_mode=backtracking, display=True)
    # print(sol)


def rastrigin_testing():
    r = Descent(rastrigin)

    x0 = np.array([-0.1, 0.5])
    eq = r.descend(x0, r.newton, wolfe)
    eq2 = r.BFGS(x0)

    print(eq)
    print(eq2)

# himmelblau_testing()
rastrigin_testing()
