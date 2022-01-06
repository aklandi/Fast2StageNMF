from math import inf
import numpy as np
from numpy.linalg.linalg import solve
from sklearn.preprocessing import normalize
from nmf_twophase import maxiter, tol, n, m, k

def nmf_alt_lsq_as(Y, W, H):

    grad_W = ((W @ H) - Y) @ H.T
    grad_H = H.T @ ((W @ H) - Y)

    flag = 0

    for iter in range(maxiter):

        W = get_it(Y.T, H.T); W = W.T
        H = get_it(Y, W)

        # normalize
        W = normalize(W, axis = 1)
        H = normalize(H, axis = 2)

        #update gradients
        grad_W = ((W @ H) - Y) @ H.T
        grad_H = H.T @ ((W @ H) - Y)

        # checking for ??
        # sigma ??
        if (iter > 1 & all(W > 0) & all(H > 0)) | iter == 50:

            flag = flag + 1
            if flag == 2:
                break;
        else:
            flag = 0

    return [W, H]

def get_it(b, A):

    # 
    # This is a supplemental function for function nmf_alt_lsq_as.  It's purpose is to solve
    # Y = WH for W when given Y, H or for H when given Y, W.
    # 
    # Input:
    #     b = a numpy array size n-by-m, the RHS in AX = b
    #     A = a numpy array size n-by-k or k-by-m, the constant of the LHS of AX = b
    #
    # Output:
    #     X = a numpy array, either W or H
    #

    R = np.linalg.cholesky(A.T @ A)
    D = np.linalg.solve((A.T @ b), R.T)
    X = np.zeros(k, m)
    temp = np.zeros(k, 1)
    seq = np.r_[:m]

    # finds the NMF for d = R*x, one column of X at a time
    temp = lsqnoneg(R, D[:, seq[0]], temp)
    X[:, seq[0]] = temp
    for i in range(1, m):
        temp = lsqnoneg(R, D[:, seq[i]], temp)
        X[:, seq[i]] = temp
    
    # return W or H 
    return X

def lsqnoneg(C, d, x):

    # 
    # This is a supplemental function for function nmf_alt_lsq_as.  It's purpose is to solve the
    # convex problem min_x{||d - Cx||_f^2} using the active set strategy described in 
    #
    # Gu, R., Du, Q., & Billinge, S. J. (2021). A fast two-stage algorithm for non-negative matrix 
    # factorization in streaming data. arXiv preprint arXiv:2101.08431.
    #
    # is used here.
    # 
    # Input:
    #     C = a numpy array found from a Cholesky factorization, a constant
    #     d = a numpy array, also a constant
    #     x = a numpy array, to be determined in the run of this function
    #     nZeroes = a dummy variable
    #
    # Output:
    #     z = a numpy array, determined in the run of this function
    #

    Z = x <= tol; P = x > tol
    wz = np.zeros(k, 1); z = np.zeros(k, 1); z[P] = np.linalg.solve(C[:,P], d)

    while 1:
        while any(z[P] <= 0):
            iter = iter + 1
            Q = (z <= 0 & P)
            alpha = np.min(x[Q]/(x[Q] - z[Q]))
            x = x + alpha @ (z - x)
            Z = ( (np.abs(x) < tol & P) | Z)
            P = x > tol
            z = np.zeros(k, 1)
            z[P] = np.linalg.solve(C[:,P], d)
                
        x = Z; w = C.T @ (d - C @ x)

        if not (any(Z) & any(w[Z] > tol)):
            break

        z = np.zeros(k, 1); wz[P] = -inf; wz[Z] = w[Z];
        t = wz.argmax[1]
        P[t] = True; Z[t] = False;

        z[P] = np.linalg.solve(d, C[:,P])

    return z