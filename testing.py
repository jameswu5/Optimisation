from step_selection import backtracking, wolfe
from test_functions import rosenbrock, himmelblau, ackley, rastrigin
from descent import Descent
import numpy as np


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
    Newton method doesn't work, maybe this is due to the hessian?
    """
    h = Descent(himmelblau)

    p1 = np.array([0.5, 0.5])
    p2 = np.array([-5., 5.])
    p3 = np.array([-5., -5.])
    p4 = np.array([3., -2.])

    # eq1 = h.descend(p1, h.steepest, backtracking)  # converges to 1st equilibrium
    # eq2 = h.descend(p2, h.steepest, backtracking)  # converges to 2rd equilibrium
    # eq3 = h.descend(p3, h.steepest, backtracking)  # converges to 3nd equilibrium
    # eq4 = h.descend(p4, h.steepest, backtracking)  # converges to 4th equilibrium

    # eq1 = h.descend(np.array([0., 0.]), h.newton, backtracking, display=True)

    # print(eq1)
    # print(eq2)
    # print(eq3)
    # print(eq4)

    # Strangely enough, on (-5, -5) wolfe and backtracking converges to different equilibria
    sol = h.BFGS(p3, step_selection_mode=wolfe, display=True)
    
    print(sol)


def rastrigin_testing():
    r = Descent(rastrigin)

    x0 = np.array([-0.1, 0.5])
    eq = r.descend(x0, r.newton, wolfe)
    eq2 = r.BFGS(x0)

    print(eq)
    print(eq2)

# rosenbrock_testing()
# himmelblau_testing()
# rastrigin_testing()
