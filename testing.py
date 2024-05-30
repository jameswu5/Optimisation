from line_search import steepest_descent
from test_functions import rosenbrock
import numpy as np

# Exercise 3.1 (page 63)
def exercise_3_1():
    x0 = np.array([-1.2, 1]).T

    print(steepest_descent(rosenbrock.func, rosenbrock.derivative, x0))