from step_selection import backtracking, wolfe
from test_functions import rosenbrock
from descent import Descent
import numpy as np


# Exercise 3.1 (page 63)
def exercise_3_1():
    x0 = np.array([-1.2, 1]).T
    # x0 = np.array([1.2, 1.2]).T

    r = Descent(rosenbrock)
    # sol = r.descend(x0, r.steepest, backtracking)
    # sol = r.descend(x0, r.newton, backtracking)
    sol = r.descend(x0, r.steepest, wolfe)
    print(sol)


exercise_3_1()
