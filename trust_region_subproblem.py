import numpy as np


# section 4.3
def subproblem_solve_newton(delta, B, df, lambda1, lambda0=10):
    """
    Attempt Algorithm 4.3 in the book (root-finding Newton's method).
    Safeguards from a paper (Computing a Trust region step).

    delta (float) : trust region size
    B (matrix): Some symmetric matrix, either identity, Hessian or approx
    df (func): gradient of objective function f
    lambda1: Smallest eigenvalue of matrix B
    lambda0 (float) : inital guess of lambda
    """

    n = np.size(df)
    lambda_l = lambda0

    # safeguards from paper
    lambda_s = max(-np.diagonal(B))
    lambda_low = max(0, lambda_s, np.linalg.norm(df)/delta - np.sum(B))
    lambda_up = np.linalg.norm(df)/delta + np.sum(B)

    # L L^T function output though R^T R is algorithm format
    # Termination criteria for future improvement
    for i in range(5):
        L = np.linalg.cholesky(B + lambda_l * np.identity(n))
        p = -np.linalg.inv(L.T) @ np.linalg.inv(L) @ df
        q = np.linalg.inv(L) @ p

        lambda_l += ((np.linalg.norm(p)/np.linalg.norm(q))**2
                     * (np.linalg.norm(p) - delta) / delta)

        lambda_l = max(lambda_l, lambda_low)
        lambda_l = min(lambda_l, lambda_up)
        if lambda_l <= lambda_s:
            lambda_l = max(0.001*lambda_up, np.sqrt(lambda_low*lambda_up))

        # update safeguards, lambda_s update future improvement
        if lambda_l > -lambda1 and (1/delta - 1/p < 0):
            lambda_up = min(lambda_up, lambda_l)
        else:
            lambda_low = max(lambda_low, lambda_l)
        lambda_low = max(lambda_low, lambda_s)


def subproblem_hard(delta, df, e_val, e_vec):
    """
    Resolve hard case as in page 88

    delta (float) : trust region size
    df (func): gradient of objective function f
    e_val: Eigenvalues of the matrix B in ascending order
    e_vec: Eigenvectors of the matrix B
    """
    z = e_vec[0] / np.linalg.norm(e_vec[0])

    j_index = np.where(e_val != e_val[0])[0]
    components = e_vec[:, j_index].T @ df / (e_val[j_index]-e_val[0])
    tau = np.sqrt(delta**2 - np.linalg.norm(components)**2)
    return np.dot(e_vec[:, j_index], components) + tau*z


def subproblem_solve(delta, B, df, lambda0=10):
    """
    Attempt to solve the trust region subproblem more accurately.

    delta (float) : trust region size
    B (matrix): Some symmetric matrix, either identity, Hessian or approx
    df (func): gradient of objective function f
    lambda0 (float) : inital guess of lambda
    """

    # simple case, B positive definite
    try:
        L = np.linalg.cholesky(B)
    except LinAlgError:
        pass
    else:
        p = np.linalg.inv(L.T) @ np.linalg.inv(L) @ df
        if np.linalg.norm(p) <= delta:
            return -p

    e_val, e_vec = np.linalg.eigh(B)
    e_vec_1 = e_vec[:, e_val == e_val[0]]
    if np.linalg.norm(e_vec_1.T @ df) < 1e-7:
        return subproblem_hard(delta, df, e_val, e_vec)

    return subproblem_solve_newton(delta, B, df, e_val[0], lambda0)


class LinAlgError(Exception):
    pass
