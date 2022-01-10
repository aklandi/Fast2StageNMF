import numpy as np
from time import process_time
import nmf_alt_lsq_as
import nmf_int_precor

def nmf_twophase(data, W, H):

    #
    # This function implements the two stage method described in
    # 
    # Gu, R., Du, Q., & Billinge, S. J. (2021). A fast two-stage 
    # algorithm for non-negative matrix factorization in streaming data. arXiv preprint arXiv:2101.08431.
    #
    # Original program in Matlab on GitHub: ... 
    # This adaptation is done in Python 3.8.10, author Amanda Landi, amanda.k.landi@gmail.com.
    #
    # Input: 
    #     data = a data point in a sequence, expected to be a numpy array, size n-by-m
    #     W = an initial dictionary matrix for the NMF factorization of data, expected
    #         to be a numpy array, size n-by-k
    #     H = an initial assignment matrix for the NMF factorization of data, expected 
    #         to be a numpy array, size k-by-m
    #
    # Output:
    #     W = the final result for two stage NMF dictionary matrix
    #     H = the final result for two stage NMF assignment matrix
    #     time = the cpu time it took to run two stage NMF for this data point
    #     rel_residual = the relative Frobenius residual that compares the final W*H with the original data
    #

    # define global variables
    global n, m; [n, m] = data.shape
    global k; k = H.shape[0]
    global maxiter; maxiter = 200
    global tol; tol = 1e-6
    global rho; rho = 1e-3

    data = np.maximum(data, np.zeros([n,m]))

    elapse_start_phase1 = process_time()
    # Phase 1
    [W, H] = nmf_alt_lsq_as(data, W, H)
    elapse_stop_phase1 = process_time()
    time_phase1 = elapse_stop_phase1 - elapse_start_phase1

    W = np.max(W, rho)
    H = np.max(H, rho)

    elapse_start_phase2 = process_time()
    # Phase 2
    [W, H] = nmf_int_precor(data, W, H)
    elapse_stop_phase2 = process_time()
    time_phase2 = elapse_stop_phase2 - elapse_start_phase2
    time = time_phase1 + time_phase2

    rel_residual = np.linalg.norm(data - (W @ H), ord = "fro")/np.linalg.norm(data, ord = "fro")

    return [W, H, time, rel_residual]
