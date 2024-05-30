from line_search import backtracking_line_search
from test_functions import rosenbrock
from descent import Descent
import numpy as np

# Exercise 3.1 (page 63)
def exercise_3_1():
    x0 = np.array([-1.2, 1]).T

    r = Descent(rosenbrock)
    sol = r.descend(x0, r.steepest, backtracking_line_search)
    print(sol)

exercise_3_1()