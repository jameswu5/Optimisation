import numpy as np
from modification_methods import symmetric_indefinite_factorization
"""
subproblem_solve: Obtain a better step in each trust region iteration compared 
to trm.py

The algorithm is still suboptimal compared to the literature, but
this will do for now (especially Newton's method).
section 4.3


"""


def subproblem_solve_newton(df, B, delta, lambda1, lambda0, e_vec0):
    """
    Attempt Algorithm 4.3 in the book (root-finding Newton's method).
    Safeguards from a paper (Computing a Trust region step).
    Safeguards not complete, in particular from 
    Termination criteria not included here.

    df (func): gradient of objective function f
    B (matrix): Some symmetric matrix, either identity, Hessian or approx
    delta (float) : trust region size
    lambda1(float): Smallest eigenvalue of matrix B
    lambda0 (float) : inital guess of lambda
    e_vec0(array): Eigenvector of B corresponding to lambda1
    """

    n = len(B)
    lambda_l = lambda0

    # safeguards from paper

    lambda_s = max(-np.diagonal(B))
    lambda_low = max(0, lambda_s, np.linalg.norm(df)/delta
                     - np.linalg.norm(B, ord=1))
    lambda_up = np.linalg.norm(df)/delta + np.linalg.norm(B, ord=1)

    # print("lambda0")
    # print(lambda1, lambda_l, lambda_low, lambda_s, lambda_up)
    # print()
    # print(B)
    # print(np.linalg.eigvalsh(B))
    # print(B + lambda_l * np.identity(n))
    # print(np.linalg.cholesky(B + lambda_l * np.identity(n)))

    # L L^T function output though R^T R is algorithm format
    # Note increasing number of iterations here barely makes a difference in the iteration plots
    for _ in range(5):

        try:
            L = np.linalg.cholesky(B + lambda_l * np.identity(n))
        except np.linalg.LinAlgError:
            pass
            B_posdef = 0
        else:
            p = -np.linalg.inv(L.T) @ np.linalg.inv(L) @ df
            B_posdef = 1
            
        # update safeguards
        # update lambda_s via (3.9) in the paper for mastery in future
        if lambda_l > -lambda1 and (1/delta - 1/np.linalg.norm(p) < 0):
            lambda_up = min(lambda_up, lambda_l)
            z = e_vec0 / np.linalg.norm(e_vec0)
            lambda_s = max(lambda_s, lambda_l - np.linalg.norm(L.T@z)**2)
        else:
            lambda_low = max(lambda_low, lambda_l)

        lambda_low = max(lambda_low, lambda_s)

        if B_posdef:
            q = np.linalg.inv(L) @ p
            lambda_l += ((np.linalg.norm(p)/np.linalg.norm(q))**2
                         * (np.linalg.norm(p) - delta) / delta)
        else:
            lambda_l = lambda_s

        # safeguard
        if lambda_l <= lambda_s:
            lambda_l = max(0.001*lambda_up, np.sqrt(lambda_low*lambda_up))
        lambda_l = max(lambda_l, lambda_low)
        lambda_l = min(lambda_l, lambda_up)
        

    # print("lambda1")
    # print(lambda1, lambda_l, lambda_low, lambda_s, lambda_up)
    # print()
    # print(B)
    # print(np.linalg.eigvalsh(B))
    # print(B + lambda_l * np.identity(n))
    # print(np.linalg.cholesky(B + lambda_l * np.identity(n)))
    # print()
    # print()

    # lambda_l < -lambda1 possible at end
    return -np.linalg.inv(B + lambda_l * np.identity(n)) @ df


def subproblem_hard(df, e_val, e_vec, delta):
    """
    Resolve hard case as in page 88

    df (func): gradient of objective function f
    e_val(array): Eigenvalues of the matrix B in ascending order
    e_vec(array): Eigenvectors of the matrix
    delta (float) : trust region size
    """
    z = e_vec[0] / np.linalg.norm(e_vec[0])

    j_index = np.where(e_val != e_val[0])[0]
    components = e_vec[:, j_index].T @ df / (e_val[j_index]-e_val[0])
    tau = np.sqrt(delta**2 - np.linalg.norm(components)**2)
    return np.dot(e_vec[:, j_index], components) + tau*z


def subproblem_solve(df, B, delta):
    """
    Attempt to solve the trust region subproblem more accurately.

    df (func): gradient of objective function f
    B (matrix): Some symmetric matrix, either identity, Hessian or approx
    delta (float) : trust region size
    """

    # simple case, B positive definite
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        pass
    else:
        p = np.linalg.inv(L.T) @ np.linalg.inv(L) @ df
        if np.linalg.norm(p) <= delta:
            return -p

    e_val, e_vec = np.linalg.eigh(B)
    e_vec_1 = e_vec[:, e_val == e_val[0]]
    if np.linalg.norm(e_vec_1.T @ df) < len(B)*1e-12:
        return subproblem_hard(df, e_val, e_vec, delta)

    return subproblem_solve_newton(df, B, delta, e_val[0], 3*abs(e_val[0]), e_vec[0])

# copied from trm
def trm_subproblem(f, grad_f, hess_f, x0, delta0, delta_max, eta, iter_time, tolerance):
    x = x0
    delta = delta0

    for k in range(iter_time):
        p = subproblem_solve(grad_f(x), hess_f(x), delta)
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


def SR1_algo(sub_method, f, df, x0, delta0, eta, iter_time, r=1e-8, tolerance=1e-8):
    """
    Algorithm 6.2
    SR1 trust region method with approximated Hessian (page 146)

    sub_method(func): function used to solve trust region subproblem
    f(func): objective function to minimise
    df(array): Gradient of f
    x0(array): starting point
    delta0(float): initial trust region radius
    eta(float): number in (0, 1e-3)
    iter_time(int): max number of iterations
    r(float): 0<r<1, say 1e-8 (suggested in book page 146)
    tolerance(float): acceptable level of error
    """

    x = x0
    xs = [np.copy(x)]
    delta = delta0
    B = np.eye(len(x0))

    for k in range(iter_time):
        if np.linalg.norm(df(x)) < tolerance:
            return xs
        
        sk = sub_method(df(x), B, delta)
        yk = df(x + sk) - df(x)
        ared = f(x) - f(x + sk)
        pred = -df(x).T @ sk + 0.5 * sk.T @ B @ sk
        rho = ared/pred

        # print()
        # print("iter", k)

        if rho > eta:
            x += sk
        xs.append(np.copy(x))

        if rho > 0.75:
            if np.linalg.norm(sk) > 0.8*delta:
                delta *= 2
        elif 0.1 <= rho <= 0.75:
            pass
        else:
            delta *= 0.5

        if abs(sk @ (yk-B@sk)) >= r*np.linalg.norm(sk)*np.linalg.norm(yk-B@sk):
            B += np.outer(yk - B@sk, yk - B@sk)/((yk-B@sk) @ sk)

    print(x, df(x), B)
    raise ConvergenceError("Fail to find a smooth local minimum")


def SR1_trm(sub_method, f, df, x0, delta0, eta, iter_time, r=1e-8, tolerance=1e-8):
    """
    Algorithm 6.2
    SR1 trust region method with approximated Hessian (page 146)

    sub_method(func): function used to solve trust region subproblem
    f(func): objective function to minimise
    df(array): Gradient of f
    x0(array): starting point
    delta0(float): initial trust region radius
    eta(float): number in (0, 1e-3)
    iter_time(int): max number of iterations
    r(float): 0<r<1, say 1e-8 (suggested in book page 146)
    tolerance(float): acceptable level of error
    """
    xlis = SR1_algo(sub_method, f, df, x0, delta0, eta, iter_time, r=1e-8, tolerance=1e-8)
    return xlis[-1]

class ConvergenceError(Exception):
    pass
