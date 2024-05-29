import numpy as np

# Algorithm 3.1 (page 37)
def backtracking_line_search(f, df, xk, pk, rho, c, alpha_bar):
    """
    f (func): objective function
    df (func): gradient of f
    xk (array): current point
    pk (array): direction
    rho (float): reduction factor
    c (float): sufficient decrease condition parameter
    alpha_bar (float): initial step length
    """

    alpha = alpha_bar
    while f(xk + alpha * pk) > f(xk) + c * alpha * df(xk).T @ pk:
        alpha *= rho
    return alpha


def descent(f, df, x0, dir_func, tolerance=1e-8, max_iterations=5000):
    """
    f (func): objective function
    df (func): gradient of f
    x0 (array): initial point
    dir_func (func): function obtaining descent direction
    tolerance (float): acceptable level of error
    max_iterations (int): maximum iterations
    """

    x = x0
    rho = 0.5
    c = 0.5
    alpha_bar = 1

    for _ in range(max_iterations):
        # check if it's a local minimum by checking gradient
        if np.linalg.norm(df(x)) < tolerance:
            return x

        # descent direction
        p = dir_func(x)

        alpha = backtracking_line_search(f, df, x, p, rho, c, alpha_bar)
        x += alpha * p

    raise ConvergenceError("Unable to find a local minimum.")


def steepest_descent(f, df, x0):
    def descent_direction(x):
        return -df(x)
    return descent(f, df, x0, descent_direction)


class ConvergenceError(Exception):
    pass



# Exercise 3.1 (page 63)
def exercise_3_1():
    x0 = np.array([-1.2, 1]).T

    # here x is a np array of two elements
    def rosenbrock(x):
        x1, x2 = x
        return 100 * (x2 - x1**2)**2 + (1-x1)**2

    def d_rosenbrock(x):
        x1, x2 = x
        return np.array([
            -400*x1*(x2-x1**2) - 2*(1-x1),
            200*(x2-x1**2)
        ])

    print(steepest_descent(rosenbrock, d_rosenbrock, x0))



# TODO: Newton method, Quasi-newton method