import numpy as np


"""
Every line search algorithm takes the following parameters:
f (func): objective function
df (func): gradient of f
x (array): current point
p (array): current direction
"""

RHO = 0.5
C = 0.5
ALPHA_BAR = 1

# Algorithm 3.1 (page 37)
def backtracking_line_search(f, df, xk, pk):
    """
    f (func): objective function
    df (func): gradient of f
    xk (array): current point
    pk (array): direction
    rho (float): reduction factor
    c (float): sufficient decrease condition parameter
    alpha_bar (float): initial step length
    """

    alpha = ALPHA_BAR
    while f(xk + alpha * pk) > f(xk) + C * alpha * df(xk).T @ pk:
        alpha *= RHO
    return alpha



# TODO: Newton method, Quasi-newton method