import numpy as np


"""
Every line search algorithm takes the following parameters:
f (func): objective function
df (func): gradient of f
x (array): current point
p (array): current direction
"""

# Algorithm 3.1 (page 37)
RHO = 0.5       # reduction factor
C = 0.5         # sufficient descrease condition parameter
ALPHA_BAR = 1   # initial step length

def backtracking(f, df, x, p):
    alpha = ALPHA_BAR
    while f(x + alpha * p) > f(x) + C * alpha * df(x).T @ p:
        alpha *= RHO
    return alpha


# Algorithm 3.5 (page 60)
MAX_ITERATIONS = 1000
C1 = 1e-4
C2 = 0.9

def wolfe(f, df, x, p):
    phi = lambda alpha: f(x + alpha * p)
    dphi = lambda alpha: p.T @ df(x + alpha * p)

    alpha_max = 2 # this is an arbitrary choices
    prev_alpha = 0
    alpha = alpha_max / 2 # this is an arbitrary choice

    # Precompute reusable values
    phi_0 = phi(0)
    dphi_0 = dphi(0)

    def zoom(alpha_low, alpha_high):
        for _ in range(MAX_ITERATIONS):
            # assume it's halfway for now. it's incorrect and I will change later.
            alpha_j = (alpha_low + alpha_high) / 2

            phi_j = phi(alpha_j)
            if phi_j > phi_0 + C1 * alpha_j * dphi_0 or phi_j >= phi(alpha_low):
                alpha_high = alpha_j
                continue

            dphi_j = dphi(alpha_j)
            if abs(dphi_j) <= -C2 * dphi_0:
                return alpha_j

            if dphi_j * (alpha_high - alpha_low) >= 0:
                alpha_high = alpha_low
            alpha_low = alpha_j

        raise Exception()
    
    for i in range(MAX_ITERATIONS):
        if phi(alpha) > phi_0 + C1 * alpha * dphi_0 or (phi(alpha) >= phi(prev_alpha) and i > 0):
            return zoom(prev_alpha, alpha)
        
        if abs(dphi(alpha)) <= -C2 * phi_0:
            return alpha
        
        if dphi(alpha) >= 0:
            return zoom(alpha, prev_alpha)

        prev_alpha, alpha = alpha, (alpha + alpha_max) / 2 # I choose it to be halfway through

    raise Exception()