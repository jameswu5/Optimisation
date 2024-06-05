import autograd.numpy as np
from autograd import grad, hessian


def cauchy(f, x, delta):
    grad_f = grad(f)
    hess_f = hessian(f)
    g = grad_f(x)
    B = hess_f(x)

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


def trm_cauchy(f, x0, delta0, delta_max, eta, iter_time, tolerance):
    # Initialize
    x = x0
    delta = delta0
    grad_f = grad(f)
    hess_f = hessian(f)

    # Iterate using cauchy point calculation
    for k in range(iter_time):
        p = cauchy(f, x, delta)
        ar = f(x) - f(x + p)
        pr = -np.dot(grad_f(x), p) - 0.5 * np.dot(p.T, np.dot(hess_f(x), p))
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


def dogleg(f, x, delta):
    grad_f = grad(f)
    hess_f = hessian(f)
    g = grad_f(x)
    B = hess_f(x)

    # Calculate Cauchy point
    p_u = cauchy(f, x, delta)

    # Compute Newton direction
    try:
        p_b = - np.linalg.solve(B, g)
        if np.any(np.linalg.eigvals(B) <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        p_b = p_u

    # Solve depending on the trust region constraint
    if np.linalg.norm(p_b) <= delta:
        return p_b
    elif np.linalg.norm(g) > delta:
        return - g * delta / np.linalg.norm(g)
    else:  # Solve for intermediate values of delta
        u = p_u
        v = p_b - p_u
        tau = (-np.dot(u, v) + np.sqrt(np.dot(u, v)**2 + np.dot(v, v) * (delta**2 - np.dot(u, u)))) / np.dot(v, v)
        if 0 <= tau <= 1:
            return tau * u
        elif 1 <= tau <= 2:
            return u + (tau - 1) * v


def trm_dogleg(f, x0, delta0, delta_max, eta, iter_time, tolerance):
    x = x0
    delta = delta0
    grad_f = grad(f)
    hess_f = hessian(f)

    for k in range(iter_time):
        p = dogleg(f, x, delta)
        ar = f(x) - f(x + p)
        pr = -np.dot(grad_f(x), p) - 0.5 * np.dot(p.T, np.dot(hess_f(x), p))
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


def subspace(f, x, delta):
    grad_f = grad(f)
    hess_f = hessian(f)
    g = grad_f(x)
    B = hess_f(x)
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
        alpha = -lambda1 * 1.5

        try:
            p_b = -np.linalg.solve(B + alpha * np.eye(len(B)), g)
        except np.linalg.LinAlgError:
            p_b = -g

        # Calculate the vector v
        if np.linalg.norm(p_b) <= delta:
            p = p_b
        else:
            v = delta * (p_b / np.linalg.norm(p_b)) - p_b
            v = v - np.dot(v, np.linalg.solve(B + alpha * np.eye(len(B)), g)) * np.linalg.solve(B + alpha * np.eye(len(B)), g)

            # Update the step p
            p = p_b + v
            if np.linalg.norm(p) > delta:
                p = (delta / np.linalg.norm(p)) * p

    # Define the step to be the Cauchy point
    elif p is None and np.all(np.linalg.eigvals(B) >= 0) and np.any(np.linalg.eigvals(B) == 0):
        return cauchy(f, x, delta)

    if p is None:
        p_u = - (np.dot(g.T, g) / np.dot(g.T, np.dot(B, g))) * g
        u = p_u
        v = p_b - p_u
        tau = (-np.dot(u, v) + np.sqrt(np.dot(u, v)**2 + np.dot(v, v) * (delta**2 - np.dot(u, u)))) / np.dot(v, v)
        if 0 <= tau <= 1:
            return tau * u
        elif 1 <= tau <= 2:
            return u + (tau - 1) * v

    if p is None:
        p = -g * min(1, delta / np.linalg.norm(g))

    return p


def trm_subspace(f, x0, delta0, delta_max, eta, iter_time, tolerance):
    x = x0
    delta = delta0
    grad_f = grad(f)
    hess_f = hessian(f)

    for k in range(iter_time):
        p = subspace(f, x, delta)
        if p is None or np.linalg.norm(p) > delta:
            p = p * (delta / np.linalg.norm(p))
        ar = f(x) - f(x + p)
        pr = -np.dot(grad_f(x), p) - 0.5 * np.dot(p.T, np.dot(hess_f(x), p))
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
