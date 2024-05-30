from test_functions import TestFunction
import numpy as np

TOLERANCE = 1e-8
MAX_ITERATIONS = 5000


class Descent:
    def __init__(self, function: TestFunction):
        self.f = function.func
        self.df = function.derivative
        self.hf = function.hessian

    def descend(self, x0, descent_mode, line_search, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS, display=False):
        """
        x0 (array): initial point
        descent_mode (func): function obtaining descent direction
        line_search (func): function performing the line search
        tolerance (float): acceptable level of error
        max_iterations (int): maximum iterations
        """

        x = x0

        for _ in range(max_iterations):
            # check if it's a local minimum by checking gradient
            if np.linalg.norm(self.df(x)) < tolerance:
                return x

            # descent direction
            p = descent_mode(x)

            alpha = line_search(self.f, self.df, x, p)

            x += alpha * p

            if display:
                print(x)

        raise ConvergenceError("Unable to find a local minimum.")
    

    # Descent modes
    def steepest(self, x):
        return -self.df(x)
    
    def newton(self, x):
        raise NotImplemented
    
    def quasi_newton(self, x):
        raise NotImplemented



class ConvergenceError(Exception):
    pass
