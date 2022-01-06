from math import inf, log
import numpy as np
from numpy.linalg import norm, inv, cholesky, solve
from scipy.linalg import lu
from nmf_twophase import maxiter, tol, n, m, k

def nmf_int_precor(data, W, H):

    # initiate rho
    rho = tol
    # initiate eta
    eta = 0

    # initiate r and s
    grad_W = ((W @ H) - data) @ H.T
    grad_H = H.T @ ((W @ H) - data)
    r = np.ones(k*n, 1)*np.max(np.abs(grad_W))
    s = np.ones(k*m, 1)*np.maximum(np.abs(grad_H))

    # initiate mu and sigma
    mu = inf
    sigma = inf

    for i in range(maxiter):

        # set up for tolerance check of E(w_k, h_k, r_k, s_k) in algorithm 2
        grad_W = ((W @ H) - data) @ H.T
        grad_H = H.T @ ((W @ H) - data)

        gra_W = grad_W.reshape((k*n, 1))
        gra_H = grad_H.reshape((k*m, 1))

        w = W.reshape((k*n, 1))
        h = H.reshape((k*m, 1))

        temp1 = np.vstack((gra_W - r,gra_H - s))
        temp2 = np.vstack((w * r, h * s))

        # tolerance check
        if np.max([norm(temp1), norm(temp2)]) < tol:
            break

        # initiate mu, eta
        mu = 0
        nu = 0
        eta = sigma < 1e-2

        # Finding Q2 in equation (5)
        if i == 1:
            Q2 = np.zeros(k*m, 1)

        for j in range(m):
            
            indx = np.r_[((j-1)*k + 1):(j*k)]
            Q2[indx, indx] = (W.T @ W) + np.diag(s[indx]/(h[indx] + rho))

        # Finding CP in equation (?)
        if i == 1:
            CP1i = np.zeros(m*k, n*k); Ri = np.zeros(n*k, k); Rit = Ri;

        Rx = r/w

        for j in range(n):
            
            indx = np.r_[((j-1)*k + 1):(j*k)]
            Q1 = inv(cholesky( (H @ H.T) + np.diag(Rx[indx] + rho) ))

            Ri[indx,:] = Q1
            Rit[indx,:] = Q1.T
            temp1 = W[j,:]
            temp2 = (Q1.T.reshape((k*k,1)) @ (eta*(W @ H - data)[j,:])).reshape((k,m*k))
            F = np.repeat(temp1.T, [m, k]) * np.repeat(Q1.T @ H, [k, 1]).reshape((k, m*k)).T + temp2.T
            CP1i[:, indx] = F

        # prediction step
        
        beq1 = -gra_W
        beq2 = -gra_H
        reshape_beq1 = (np.repeat(beq1.reshape((k, n)), [k, 1])).reshape((k, k*n)).T
        P1ib1 = np.sum(Rit * (reshape_beq1), axis = 1)
        [P, L, U] = lu(Q2 - (CP1i @ CP1i.T))
        solu2 = solve(solve(beq2 - (CP1i @ P1ib1), L), U)
        reshape_diff = (np.repeat((P1ib1 - (CP1i.T @ solu2)).reshape((k, n)), [k, 1])).reshape((k, k*n)).T
        solu1 = np.sum(Ri * (reshape_diff), axis = 1)
        dr = ((mu/w) - r) - (r * solu1/w)
        ds = ((nu/h) - s) - (s * solu2/h)
        solutions = -np.vstack([solu1, solu2])/np.vstack([w, h])
        derivs = -np.vstack([dr, ds])/np.vstack([r, s])
        step1 = np.min([1/np.max(solutions), 1])
        step2 = np.min([1/np.max(derivs), 1])
        mu_aff = ( (np.vstack([w, h]) + step1*np.vstack([w,h])).T @ (np.vstack([r,s]) + step2*np.vstack([dr,ds])))/(n*k + m*k)
        mu = (np.vstack([w,h]).T @ np.vstack([r,s]))/(n*k + m*k)
        sigma = np.min([(mu_aff/mu)^3, 0.99])

        aff_1 = solu1 * dr; aff_2 = solu2 * ds;

        beq1 = (mu*sigma - aff_1)/(w - gra_W);
        beq2 = (mu*sigma - aff_2)/(h - gra_H);

        reshape_beq1 = (np.repeat(beq1.reshape((k, n)), [k, 1])).reshape((k, k*n)).T
        P1ib1 = np.sum(Rit * (reshape_beq1), axis = 1)
        solu2 = solve(solve(beq2 - (CP1i @ P1ib1), L), U)
        reshape_diff = (np.repeat((P1ib1 - (CP1i.T @ solu2)).reshape((k, n)), [k, 1])).reshape((k, k*n)).T
        solu1 = np.sum(Ri * (reshape_diff), axis = 1)
        dr = (((mu*sigma - aff_1)/w) - r) - (r * solu1/w)
        ds = (((mu*sigma - aff_2)/h) - s) - (s * solu2/h)

        # update step-sizes
        solutions = -np.vstack([solu1, solu2])/np.vstack([w, h])
        derivs = -np.vstack([dr, ds])/np.vstack([r, s])
        step1 = np.min([0.9/np.max(solutions), 1])
        step2 = np.min([0.9/np.max(derivs), 1])

        # update the gradients for W and H
        dW = solu1.reshape((k,n)).T
        dH = solu2.reshape((k,m))

        # update W, H, r, and s using gradient update
        W1 = W + dW*step1
        H1 = H + dW*step1
        r1 = r + dr*step2
        s1 = s + ds*step2

        # calculate phi(W, H)
        logW = log(W); logH = log(H)
        phi = (0.5)*(norm(W*H - data, ord = 'fro')) - mu*sigma*np.sum(logW, axis = None) - mu*sigma*np.sum(logH, axis = None)

        # calculate grad_phi(W,H)
        grad_phi = np.vstack([1/(grad_W.flatten() - mu*sigma*W.flatten()), 1/(grad_H.flatten() - mu*sigma*H.flatten())])

        d0 = np.vstack([dW.flatten(), dH.flatten()])
        if ((grad_phi.T @ d0) >= 0):
            continue

        W = W1 
        H = H1
        r = r1 
        s = s1

    return [W, H]