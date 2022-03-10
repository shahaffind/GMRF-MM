import numpy as np
from scipy import stats


def vi(k1, a1, k2, a2):
    a1 = a1.astype(np.int)
    a2 = a2.astype(np.int)

    n = a1.shape[0]
    if a2.shape[0] != n:
        raise ValueError("len(a1) != len(a2)")

    p1 = np.zeros(k1)
    p2 = np.zeros(k2)
    p = np.zeros([k1, k2])

    for i in range(n):
        p1[a1[i]] += 1
        p2[a2[i]] += 1
        p[a1[i], a2[i]] += 1

    p1 /= n
    p2 /= n
    p /= n

    H1 = stats.entropy(p1)
    H2 = stats.entropy(p2)

    I = 0.0
    for j in range(k2):
        for i in range(k1):
            pi = p1[i]
            pj = p2[j]
            pij = p[i,j]
            if pij > 0.0:
                I += pij * np.log(pij / (pi * pj))

    return H1 + H2 - 2*I


def vi_soft(k1, a1, k2, W):
    a1 = a1.astype(np.int)

    n = a1.shape[0]
    if W.shape[0] != n:
        raise ValueError("len(a1) != len(a2)")

    p1 = np.zeros(k1)
    p2 = W.sum(axis=0)
    p = np.zeros([k1, k2])

    for i in range(n):
        p1[a1[i]] += 1
        for j in range(k2):
            p[a1[i], j] += W[i, j]

    p1 /= n
    p2 /= n
    p /= n

    H1 = stats.entropy(p1)
    H2 = stats.entropy(p2)

    I = 0.0
    for j in range(k2):
        for i in range(k1):
            pi = p1[i]
            pj = p2[j]
            pij = p[i,j]
            if pij > 0.0:
                I += pij * np.log(pij / (pi * pj))

    return H1 + H2 - 2*I

