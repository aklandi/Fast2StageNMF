from math import inf
import numpy as np
from numpy.linalg import norm, inv, cholesky, solve
from scipy.linalg import lu
from nmf_twophase import maxiter, tol, n, m, k

def nmf_int_precor(data, W, H):

    # initiate eta
    eta = 0

    # initiate r and s
    grad_W = ((W @ H) - data) @ H.T
    grad_H = W.T @ ((W @ H) - data)
    r = np.ones((k*n, 1))*np.max(np.abs(grad_W))
    s = np.ones((k*m, 1))*np.max(np.abs(grad_H))

    # initiate mu and sigma
    mu = inf
    sigma = inf

    for i in range(maxiter):

        # set up for tolerance check of E(w_k, h_k, r_k, s_k) in algorithm 2
        grad_W = ((W @ H) - data) @ H.T; gra_W = grad_W.reshape((k*n, 1));
        grad_H = W.T @ ((W @ H) - data); gra_H = grad_H.reshape((k*m, 1))
        w = W.reshape((k*n, 1)); h = H.reshape((k*m, 1))

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
        if i == 0:
            Q2 = np.zeros((k*m, k*m))

        for j in range(m):
            
            indx = np.r_[j*k:(j*k+k)]
            values = s[indx]/(h[indx] + rho); D = np.diagflat(values)
            Q2[j*k:(j*k+k), j*k:(j*k+k)] = (W.T @ W) + D

        # Finding CP in equation (?)
        if i == 0:
            CP1i = np.zeros((m*k, n*k)); Ri = np.zeros((n*k, k)); Rit = Ri;

        Rx = r/(w + rho)

        for j in range(n):
            
            indx = np.r_[j*k:(j*k+k)]
            values = Rx[indx] + rho; D = np.diagflat(values)
            Q1 = inv(cholesky( (H @ H.T) + D ))

            Ri[indx,:] = Q1; Rit[indx,:] = Q1.T
            temp1 = W[j,:].reshape((1,k)); 
            temp2 = ((Q1.T.reshape((k*k,1))) @ (eta*(W @ H - data)[j,:].reshape((1,m)))).reshape((k,m*k))
            F = np.tile(temp1.T, (m, k)) * np.tile((Q1.T @ H), (k, 1)).T.reshape((k, k*m)).T + temp2.T
            CP1i[:, indx] = F


        # prediction step
        
        beq1 = -gra_W
        beq2 = -gra_H
        reshape_beq1 = (np.tile(beq1.reshape((k, n)), (k, 1))).reshape((k, k*n)).T
        P1ib1 = np.sum(Ri * (reshape_beq1), axis = 1)
        [P, L, U] = lu(Q2 - (CP1i @ CP1i.T))
        CP = (CP1i @ P1ib1).reshape((k*m, 1))
        g, _, _, _ = lstsq(beq2 - CP, L, rcond = None)
        solu2, _, _, _ = lstsq(g.T, U, rcond = None); solu2 = solu2.T
        reshape_diff = np.tile(P1ib1.reshape((n*k, 1)) - (CP1i.T @ solu2), (1, k))
        solu1 = np.sum((Ri * reshape_diff), axis = 1); solu1 = solu1.reshape((n*k, 1))
        dr = ((mu/w) - r) - (r * solu1/w)
        ds = ((nu/h) - s) - (s * solu2/h)

        solutions = -np.vstack([solu1, solu2])/np.vstack([w, h])
        derivs = -np.vstack([dr, ds])/np.vstack([r, s])
        step1 = np.min([1/np.max(solutions), 1])
        step2 = np.min([1/np.max(derivs), 1])
        mu_aff = ( (np.vstack([w, h]) + step1*np.vstack([w,h])).T @ (np.vstack([r,s]) + step2*np.vstack([dr,ds])))/(n*k + m*k)
        mu = (np.vstack([w,h]).T @ np.vstack([r,s]))/(n*k + m*k)
        sigma = np.min([((mu_aff/mu)**3), 0.99])

        aff_1 = solu1 * dr; aff_2 = solu2 * ds;

        beq1 = (mu*sigma - aff_1)/(w - gra_W);
        beq2 = (mu*sigma - aff_2)/(h - gra_H);

        reshape_beq1 = (np.tile(beq1.reshape((k, n)), (k, 1))).reshape((k, k*n)).T
        P1ib1 = np.sum(Ri * (reshape_beq1), axis = 1)
        [P, L, U] = lu(Q2 - (CP1i @ CP1i.T))
        CP = (CP1i @ P1ib1).reshape((k*m, 1))
        g, _, _, _ = lstsq(beq2 - CP, L, rcond = None)
        solu2, _, _, _ = lstsq(g.T, U, rcond = None); solu2 = solu2.T
        reshape_diff = np.tile(P1ib1.reshape((n*k, 1)) - (CP1i.T @ solu2), (1, k))
        solu1 = np.sum((Ri * reshape_diff), axis = 1); solu1 = solu1.reshape((n*k, 1))
        dr = ((mu/w) - r) - (r * solu1/w)
        ds = ((nu/h) - s) - (s * solu2/h)

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
        H1 = H + dH*step1
        r1 = r + dr*step2
        s1 = s + ds*step2

        # calculate phi(W, H)
        logW = np.log(W); logH = np.log(H)
        phi = (0.5)*(norm(W @ H - data, ord = 'fro')) - mu*sigma*np.sum(logW, axis = None) - mu*sigma*np.sum(logH, axis = None)

        # calculate grad_phi(W,H)
        grad_phi = np.vstack([1/(gra_W - mu*sigma*w), 1/(gra_H - mu*sigma*h)])

        d0 = np.vstack([dW.reshape((k*n, 1)), dH.reshape((k*m, 1))])
        if ((grad_phi.T @ d0) >= 0):
            continue

        W = np.maximum(W1, rho)
        H = np.maximum(H1, rho)
        r = r1 
        s = s1

    return [W, H]