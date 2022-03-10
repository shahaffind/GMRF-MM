import numpy as np
import warnings


class Gaussian:

    def __init__(self, use_means=True, Q=None, mu=None):
        self.use_means = use_means
        self.Q = Q.copy() if Q is not None else None
        self.mu = mu.copy() if mu is not None else None

    def log_prob(self, X):
        n, d = X.shape

        if self.use_means:
            Z = X - self.mu
        else:
            Z = X

        pi_norm = - .5 * d * np.log(2 * np.pi)
        log_det_q = .5 * np.linalg.slogdet(self.Q)[1]

        zq = np.dot(Z, self.Q)
        zqz = np.multiply(zq, Z).sum(axis=1)

        m_dist = -.5 * zqz

        return pi_norm + log_det_q + m_dist

    def update_from_weights(self, X, W_j, Q_estimator, warm_start=False):
        S, m = self.get_sufficient_statistics(X, W_j)

        if Q_estimator is None:  # use normal MLE
            self.Q = np.linalg.inv(S)
            if self.use_means:
                self.mu = m
            return

        # normalizing S to have 1's in the diag
        sqrt_vars = np.sqrt(np.diag(S)).reshape(-1, 1)
        inv_sqrt_vars = (sqrt_vars ** -1).reshape(-1, 1)

        normed_S = inv_sqrt_vars * S * inv_sqrt_vars.T
        if warm_start:
            normed_Q = sqrt_vars * self.Q * sqrt_vars.T
        else:
            normed_Q = np.eye(S.shape[0])

        Q_est_normed = Q_estimator.fit(normed_S, normed_Q)

        self.Q = inv_sqrt_vars * Q_est_normed * inv_sqrt_vars.T
        if self.use_means:
            self.mu = m

    def get_sufficient_statistics(self, X, W):
        W_sum = W.sum()
        if np.allclose(W_sum, 0):
            # print("D", end="", flush=True)
            warnings.warn("Empty Component, randomizing", RuntimeWarning)
            W = np.random.choice([0., 1.], X.shape[0], p=[.9, .1])
            W_sum = W.sum()

        if self.use_means:
            m = (X.T * W).sum(axis=1) / W_sum
            Z = X - m
        else:
            m = None
            Z = X

        S = (Z.T * W).dot(Z) / W_sum
        S += np.eye(S.shape[0]) * 1.e-5
        S = (S + S.T) / 2

        return S, m
