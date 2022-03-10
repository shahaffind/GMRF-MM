import numpy as np
from datetime import datetime
from MixtureModel.Gaussian import Gaussian


class GMM:

    def __init__(self, K, use_means=True, warm_start=False):
        self.K = K
        self.components = None
        self.Q_estimator = None

        self.pi = np.ones(K) / K
        self.dirichlet_pi_prior = None

        self.log_likelihood_record = None

        self.use_means = use_means
        self.warm_start = warm_start

    def set_dirichlet_prior(self, p):
        if p <= 1:
            raise Exception("Illegal Dirichlet prior (p<=1)")
        self.dirichlet_pi_prior = p

    def set_Q_estimator(self, estimator):
        self.Q_estimator = estimator

    def fit(self, X, max_iters=100, verbose=False, init=True):

        self.log_likelihood_record = []

        first_iter = -1 if init else 0

        for _i in range(first_iter, max_iters):

            if verbose:
                print(f"it: {_i}", end="", flush=True)

            s = datetime.now()
            if _i == -1:  # initialize components
                W = self._first_e_step(X.shape[0])
            else:
                W = self._e_step(X)
                ll = self.log_likelihood_record[_i]
                if verbose:
                    print(f"\tlog likelihood: {ll:.3f}", end="", flush=True)
            elapsed = (datetime.now() - s).total_seconds()

            if verbose:
                print(f"\tE time: {elapsed:.2f}s", end="", flush=True)

            s = datetime.now()
            if _i == -1:
                self._m_step(X, W, False)
            else:
                self._m_step(X, W, self.warm_start)
            elapsed = (datetime.now() - s).total_seconds()

            if verbose:
                print(f"\tM time: {elapsed:.2f}s", end="\n", flush=True)

    def _first_e_step(self, N):
        self.init_components()
        labels = np.random.choice(self.K, N, replace=True)
        while len(np.unique(labels)) != self.K:
            labels = np.random.choice(self.K, N, replace=True)
        W = np.eye(self.K)[labels]
        return W

    def init_components(self):
        self.components = [Gaussian(use_means=self.use_means) for _ in range(self.K)]

    def _e_step(self, X):
        weighted_log_prob = self._get_log_weighted_probabilities(X)
        W = self._normalize_proba(weighted_log_prob)

        # complete data log-likelihood
        log_likelihood = (W * weighted_log_prob).sum()
        self.log_likelihood_record.append(log_likelihood)

        return W

    def _normalize_proba(self, weighted_log_prob):
        # log-sum-exp trick
        _max_log_p = weighted_log_prob.max(axis=1, keepdims=True)
        _d = weighted_log_prob - _max_log_p
        _exp_d_sum = np.exp(_d).sum(axis=1, keepdims=True)
        _log_W = _d - np.log(_exp_d_sum)

        return np.exp(_log_W)

    def get_proba(self, X):
        weighted_log_prob = self._get_log_weighted_probabilities(X)
        return self._normalize_proba(weighted_log_prob)

    def _m_step(self, X, W, warm_start):
        self._update_pi(W)
        self._update_components(X, W, warm_start)

    def _get_log_weighted_probabilities(self, X):
        weighted_log_prob = np.empty([X.shape[0], self.K])
        for j in range(self.K):
            weighted_log_prob[:, j] = self.components[j].log_prob(X) + np.log(self.pi[j])

        return weighted_log_prob
    
    def _update_pi(self, W):
        if self.dirichlet_pi_prior is None:
            self.pi = W.mean(axis=0)
        else:
            r = W.sum(axis=0)

            dirichlet_prior = self.dirichlet_pi_prior

            # Dirichlet distribution mode
            alpha = r + dirichlet_prior
            new_pi = (alpha - 1) / (alpha.sum() - self.K)
            self.pi = new_pi

    def _update_components(self, X, W, warm_start):
        for j, c in enumerate(self.components):
            c.update_from_weights(X, W[:, j], self.Q_estimator, warm_start)
