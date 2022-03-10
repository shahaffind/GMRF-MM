import scipy as sp
import numpy as np
from scipy import linalg


def sample_by_precision(mu, Q, m=1):
    n = Q.shape[0]
    L = linalg.cholesky(Q, lower=True)
    # single sample
    if m == 1:
        z = sp.random.multivariate_normal(sp.zeros(n), sp.eye(n))
        v = linalg.solve_triangular(L.transpose(), z)
        x = mu + v
        return x
    # multiple samples
    else:
        Z = sp.random.multivariate_normal(sp.zeros(n), sp.eye(n), m)
        X = []
        for z in Z:
            v = linalg.solve_triangular(L.transpose(), z)
            x = mu + v
            X.append(x)
        return np.array(X)
