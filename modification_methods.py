import numpy as np
from scipy.linalg import cholesky

"""
These are modification algorithms from chapter 3.4

x0 : initial point
H: the given Hessian Matrix
alpha: the step length (can differ for different iteration states)
"""


def is_positive_definite(matrix):
    """Check whther a matrix is positive definite using cholesky."""
    try:
        _ = cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def spectral_decomposition(sym_matrix):
    """Compute the spectral decomposition QLQ^T of a symmetric matrix."""
    evals, evecs = np.linalg.eigh(sym_matrix)
    Lambda = np.diag(evals)
    Q = evecs
    return Q, Lambda


# Eigenvalue Modification (page 50)
def eigenvalue_modification(A, delta):
    """
    Goal: Compute the modified version of A such that the modified one is positive definite.
    Process: Add a delta_A to A with delta_A computed using method on page 50
    Input: A symmetric matrix A, a delta which we want every eigenvalue of A + delta_A to be greater to 
    Output: A modified matrix A + delta_A
    """
    # Check whether A is positive definite already
    if is_positive_definite(A):
        return A
    else:
        Q, Lambda = spectral_decomposition(A)
        n = A.shape[0]
        tau = np.ones(n)
        eigenvalues = np.linalg.eigvalsh(A)
        for i in range(n):
            if eigenvalues[i] >= delta:
                tau[i] = 0
            else:
                tau[i] = delta - eigenvalues[i]
        tau_matrix = np.diag(tau)
        delta_A = Q @ tau_matrix @ Q.T
        return A + delta_A


# Diagonal Modification (Page 50, 51)
def diagonal_modification(A, delta):
    """Almost the same as the eigenvalue one, but quite simpler"""
    eigenvalues = np.linalg.eigvalsh(A)
    min_eval = min(eigenvalues)
    tau = max(0, delta - min_eval)
    n = A.shape[0]
    return A + tau * np.eye(n)


# Algorithm 3.3 Cholesky with added multiple of the identity
def cholesky_add_multiple(A, beta=10**-3):
    """
    An alternative way to do Cholesky that ensure success by adding a multiple of identity

    Input:
    A : symmetric matrix
    beta: A small positive constant

    Return:
    L : the cholesky factorization of modified matrix A + tau * I

    To get modified matrix A*, we simply apply L @ L.T
    """
    # Find the mininal diagonal element of A
    min_diag = np.min(np.diag(A))
    if min_diag > 0:
        tau = 0
    else:
        tau = -min_diag + beta
    while True:
        try:
            L = cholesky(A + tau * np.eye(A.shape[0]), lower=True)
            return L
        except np.linalg.LinAlgError:
            tau = max(2 * tau, beta)


def modified_symmetric_indefinite_factorization(A, delta):
    P, L, B = symmetric_indefinite_factorization(A)
    Q, Lambda = spectral_decomposition(A)
    n = A.shape[0]
    tau = np.zeros(n)
    eigenvalues = np.linalg.eigvalsh(A)
    for i in range(n):
        if eigenvalues[i] >= delta:
            tau[i] = 0
        else:
            tau[i] = delta - eigenvalues[i]
    F = np.diag(tau)
    E = P.T @ L @ F @ L.T @ P
    return A + E
