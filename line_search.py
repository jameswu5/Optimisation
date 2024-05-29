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

# TODO: Program steepest descent and Newton's method
