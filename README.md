# Fast 2-Stage NMF

In the 2021 paper "A fast 2-stage algorithm for non-negative matrix factorization in streaming data" by Gu, Du, and Billinge, a new NMF method is proposed for online data.  See original paper at https://arxiv.org/pdf/2101.08431.pdf.  I attempt to implement this method in Python.  The NMF is the solution to the problem

                                                 min_{W >= 0, H >= 0} ||WH - Y||_f^2

where Y is an n-by-m original data point, and W_{n-by-k} and H_{k-by-m} are the nonnegative factors whose product is an approximation of Y.

The files in this repository are

* nmf_twophase.py = the overall method, executing both first and second stage.  The cpu time is tracked and the fidelity of the approximation WH is measured.
* nmf_alt_ls_as.py = stage 1 - the alternating least squares with active set method
* nmf_int_precor.py = stage 2 - interior point corrector method

As of (1/6/2022), testing of the code is not yet done.  That is forthcoming.
