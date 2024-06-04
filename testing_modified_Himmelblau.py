from modification_methods import diagonal_modification
from step_selection import backtracking, wolfe
from test_functions import himmelblau
import numpy as np

# Copy from the decent.py to make some changes
from test_functions import TestFunction
from step_selection import wolfe
import numpy as np

TOLERANCE = 1e-6
MAX_ITERATIONS = 100000


class DescentModified:
    def __init__(self, function: TestFunction):
        self.f = function.func
        self.df = function.derivative
        self.hf = function.hessian

    def descend(self, x0, descent_mode, step_selection_mode, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS, return_xs=False, display=False):
        """
        x0 (array): initial point
        descent_mode (func): function obtaining descent direction
        step_selection_mode (func): function performing the line search
        tolerance (float): acceptable level of error
        max_iterations (int): maximum iterations
        """

        x = x0
        xs = [np.copy(x)]

        for i in range(max_iterations):
            # check if it's a local minimum by checking gradient
            if np.linalg.norm(self.df(x)) < tolerance:
                return xs if return_xs else x

            # descent direction
            p = descent_mode(x)

            # select step size
            alpha = step_selection_mode(self.f, self.df, x, p)

            x += alpha * p

            xs.append(np.copy(x))

            if display:
                print(f"Iteration {i+1}: {x}")

        raise ConvergenceError("Unable to find a local minimum.")


    def descend2D(self, x, y, descent_mode, step_selection_mode, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS, display=False):
        return self.descend(np.array([x, y]), descent_mode, step_selection_mode, tolerance, max_iterations, display)

    """
    These are the descent modes that you can put into the descent function.
    Each of them take a single parameter, the point x (numpy array)
    """
    def steepest(self, x):
        return -self.df(x)

    def newton(self, x):
        A = diagonal_modification(self.hf(x), 10**-3)
        return -np.linalg.inv(A) @ self.df(x)

    # Algorithm 6.1 (page 140) and exercise 3.9 (page 64)
    def BFGS(self, x0, step_selection_mode=wolfe, tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS, display=False):
        x = x0
        H = np.eye(len(x0))
        for i in range(max_iterations):
            if np.linalg.norm(self.df(x)) < tolerance:
                return x
     
            # Quasi newton (6.18) in the book
            p = -H @ self.df(x)
            alpha = step_selection_mode(self.f, self.df, x, p)
            x += alpha * p

            if display:
                print(f"Iteration {i+1}: {x}")

            # Update H using (6.17)
            s = alpha * p
            y = self.df(x) - self.df(x - alpha * p)
            assert y @ s > 0
            rho = 1 / (y.T @ s)
            I = np.eye(len(x0))
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        raise ConvergenceError("Unable to find a local minimum.")


class ConvergenceError(Exception):
    pass
# Copy ended, modified the newton


def modified_himmelblau_testing():
    """
    Introducing modified hessian to himmelblau
    """
    h = DescentModified(himmelblau)

    p1 = np.array([0.5, 0.5])
    p2 = np.array([-5., 5.])
    p3 = np.array([-5., -5.])
    p4 = np.array([3., -2.])
    eq1 = h.descend(p1, h.newton, backtracking)
    eq2 = h.descend(p2, h.newton, backtracking)
    eq3 = h.descend(p3, h.newton, backtracking)
    eq4 = h.descend(p4, h.newton, backtracking) 
    print(eq1)
    print(eq2)
    print(eq3)
    print(eq4)