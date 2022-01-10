from math import inf
import numpy as np
from numpy.linalg.linalg import solve
from sklearn.preprocessing import normalize
from nmf_twophase import maxiter, tol, n, m, k

def nmf_alt_lsq_as(Y, W, H):

    # grad_W = ((W @ H) - Y) @ H.T
    # grad_H = W.T @ ((W @ H) - Y)

    flag = 0
    # iter = 0

    for iter in range(maxiter):

        W = get_it(Y.T, H.T); W = W.T
        H = get_it(Y, W)

        # normalize
        W = normalize(W, axis = 0)
        H = normalize(H, axis = 1)

        #update gradients
        # grad_W = ((W @ H) - Y) @ H.T
        # grad_H = W.T @ ((W @ H) - Y)

        # checking for ??
        # sigma ??
        if ( (iter > 1 and np.sum(W > 0) == n*k) and (np.sum(H > 0) == k*m) or iter == 50 ):

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

    # SPD check: if eigenvalues are negative or imaginary, we initialize matrix differently
    AAT = A.T @ A
    if any(np.isreal(eigvals(AAT))) or any(eigvals(AAT) <= 0) :

        new_vals = np.random.randint(low = 1, high = 100, size = AAT.shape[0])
        q,_ = qr(AAT)
        AAT = q.T @ np.diag(new_vals) @ q
        AAT = np.maximum(AAT, rho)

    r = b.shape[1]
    R = cholesky(AAT)
    D = solve(R.T, (A.T @ b))
    X = np.zeros((k, r)); 
    # initialize x to be random, nonnegative vector
    temp = np.random.rand(k, 1)
    seq = [i for i in range(r)]

    # finds the NMF for d = R*x, one column of X at a time
    temp = lsqnoneg(R, D[:, seq[0]], temp)
    X[:, seq[0]] = temp.reshape((3,))
    for i in range(1, r):
        temp = lsqnoneg(R, D[:, seq[i]], temp)
        X[:, seq[i]] = temp.reshape((3,))
    
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

    d = d.reshape((k,1))

    Z = x <= tol; indx_Z = Z.tolist(); Z = [elem[0] for elem in indx_Z]
    P = x > tol; indx_P = P.tolist(); P = [elem[0] for elem in indx_P]
    wz = np.zeros((k, 1)); 
    z = np.zeros((k, 1)); g, _, _, _ = lstsq(C[:,P], d, rcond = None); z[P] = g;
        
    for outiter in range(20):
        for inneriter in range(10):
            
            indx = z <= 0; indx = [elem[0] for elem in indx]; Q = indx and P

            if x[Q].size == 0:
                alpha = rho
            else:
                alpha = np.min(x[Q]/(x[Q] - z[Q] + rho))
                
            x = x + alpha*(z - x)
            indx = np.abs(x) < tol; indx = [elem[0] for elem in indx]; Z = (indx and P) or Z
            P = x > tol; indx = P.tolist(); P = [elem[0] for elem in indx]
            z = np.zeros((k, 1)); g, _, _, _ = lstsq(C[:,P], d, rcond = None); z[P] = g

        x = np.array(z); w = C.T @ (d - C @ x)
        indx = w[Z] > tol; indx = [elem[0] for elem in indx]

        z = np.zeros((k, 1)); 
        wz[P] = -inf; 
        if (np.sum(Z) >  0):
            wz[Z] = w[Z]
        
        t = np.argmax(wz); P[t] = True; Z[t] = False;
        g, _, _, _ = lstsq(C[:,P], d, rcond = None); z[P] = g

    return z