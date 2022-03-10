import numpy as np
from GMRF.KnownSparsityEstimator import KnownSparsityEstimator as known_estimator


class DebiasedEstimator:

    def __init__(self, base_estimator, debiasing_estimator=None):
        self.base_estimator = base_estimator
        self.debiasing_estimator = debiasing_estimator if debiasing_estimator is not None else known_estimator(max_descent_iters=5)

    def fit(self, S, Q=None):
        Q_glasso = self.base_estimator.fit(S, Q)
        C = np.abs(Q_glasso) > 1e-8
        Q = self.debiasing_estimator.fit(S, Q_glasso, C)
        return Q
