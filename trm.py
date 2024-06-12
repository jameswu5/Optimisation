import numpy as np


def cauchy(g, B, delta):

    # Calculate p_k^s
    pks = - (delta / np.linalg.norm(g)) * g

    # Calculate tau_k
    gBg = np.dot(g.T, np.dot(B, g))
    if gBg <= 0:
        tau_k = 1
    else:
        tau_k = min(np.linalg.norm(g)**3 / (delta * gBg), 1)

    # Calculate p_k^C
    pkc = tau_k * pks

    return pkc


def trm_cauchy(f, grad_f, hess_f, x0, delta0, delta_max, eta, iter_time, tolerance):
    # Initialize
    x = x0
    delta = delta0

    # Iterate using cauchy point calculation
    for k in range(iter_time):
        p = cauchy(grad_f(x), hess_f(x), delta)
        ar = f(x) - f(x + p)
        pr = -np.dot(grad_f(x), p) - 0.5 * np.dot(p.T, np.dot(hess_f(x), p))
        # Avoid division by zero or invalid value in rho
        if pr == 0:
            rho = 0
        else:
            rho = ar / pr

        # Adjust the trust region radius
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = min(2 * delta, delta_max)
        else:
            delta = delta

        # Update the iteration point
        if rho > eta:
            x = x + p
        else:
            x = x

        # Check if the current solution closed enough to the optimal solution
        if np.linalg.norm(grad_f(x)) < tolerance:
            break

    return x


def dogleg(g, B, delta):

    # Calculate Cauchy point
    p_u = cauchy(g, B, delta)

    # Compute Newton direction
    try:
        if np.any(np.linalg.eigvals(B) <= 0):
            raise np.linalg.LinAlgError("Matrix B is not positive definite")
        p_b = -np.linalg.solve(B, g)
    except np.linalg.LinAlgError:
        p_b = p_u

    p = None

    # Solve depending on the trust region constraint
    if np.linalg.norm(p_b) <= delta:
        p = p_b
    elif np.linalg.norm(g) > delta:
        p = - g * delta / np.linalg.norm(g)
    else:  # Solve for intermediate values of delta
        u = p_u
        v = p_b - p_u
        tau = (-np.dot(u, v) + np.sqrt(np.dot(u, v)**2 + np.dot(v, v) * (delta**2 - np.dot(u, u)))) / np.dot(v, v)
        if 0 <= tau <= 1:
            p = tau * u
        elif 1 <= tau <= 2:
            p = u + (tau - 1) * v

    # Ensure p has a definite value
    if p is None:
        p = -g * min(1, delta / np.linalg.norm(g))

    # Ensure p is within the trust region
    if np.linalg.norm(p) > delta:
        p = (delta / np.linalg.norm(p)) * p

    return p

def trm_dogleg(f, grad_f, hess_f, x0, delta0, delta_max, eta, iter_time, tolerance):
    x = x0
    delta = delta0

    for k in range(iter_time):
        p = dogleg(grad_f(x), hess_f(x), delta)
        ar = f(x) - f(x + p)
        pr = -np.dot(grad_f(x), p) - 0.5 * np.dot(p.T, np.dot(hess_f(x), p))
        # Avoid division by zero or invalid value in rho
        if pr == 0:
            rho = 0
        else:
            rho = ar / pr

        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = min(2 * delta, delta_max)
        else:
            delta = delta

        if rho > eta:
            x = x + p
        else:
            x = x

        if np.linalg.norm(grad_f(x)) < tolerance:
            break

    return x


def subspace(g, B, delta):
    p = None

    # Use the subspace span[g, B^{-1}g]
    if np.all(np.linalg.eigvals(B) > 0):
        try:
            p_b = -np.linalg.solve(B, g)
        except np.linalg.LinAlgError:
            p_b = -g
        if np.linalg.norm(p_b) <= delta:
            p = p_b

    # Use the subspace span[g, (B + alpha*I)^{-1}g]
    if p is None and np.any(np.linalg.eigvals(B) < 0):
        lambda1 = np.min(np.linalg.eigvals(B))

        # Make (B + alpha*I) positive definite
        alpha = -lambda1 * np.random.uniform(1 + 1e-6, 2)

        try:
            p_b = -np.linalg.solve(B + alpha * np.eye(len(B)), g)
        except np.linalg.LinAlgError:
            p_b = -g

        # Calculate the vector v
        if np.linalg.norm(p_b) <= delta:
            # Initialize v to ensure norm condition
            v = delta * (p_b / np.linalg.norm(p_b)) - p_b

            # Adjust v to ensure v^T (B + alpha I)^{-1} g <= 0
            Bg = np.linalg.solve(B + alpha * np.eye(len(B)), g)
            v = v - np.dot(v, Bg) / np.dot(Bg, Bg) * Bg

            # Update step p
            p = p_b + v

            if np.linalg.norm(p) > delta:
                p = (delta / np.linalg.norm(p)) * p

    # Define the step to be the Cauchy point
    if p is None and np.any(np.linalg.eigvals(B) == 0) and np.all(np.linalg.eigvals(B) >= 0):
        p = cauchy(g, B, delta)

    # Ensure p has a definite value
    if p is None:
        p = -g * min(1, delta / np.linalg.norm(g))

    # Ensure p is within the trust region
    if np.linalg.norm(p) > delta:
        p = (delta / np.linalg.norm(p)) * p

    return p


def trm_subspace(f, grad_f, hess_f, x0, delta0, delta_max, eta, iter_time, tolerance):
    x = x0
    delta = delta0

    for k in range(iter_time):
        p = subspace(grad_f(x), hess_f(x), delta)
        ar = f(x) - f(x + p)
        pr = -np.dot(grad_f(x), p) - 0.5 * np.dot(p.T, np.dot(hess_f(x), p))
        # Avoid division by zero or invalid value in rho
        if pr == 0:
            rho = 0
        else:
            rho = ar / pr

        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = min(2 * delta, delta_max)
        else:
            delta = delta

        if rho > eta:
            x = x + p

        if np.linalg.norm(grad_f(x)) < tolerance:
            break

    return x
