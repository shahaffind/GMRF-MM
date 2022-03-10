import numpy as np


class KnownSparsityEstimator:

    def __init__(self,
                 C=None,
                 tikhonov=0.0,
                 armijo_alpha=1.,
                 max_descent_iters=10, descent_tol=1e-5,
                 max_pcg_iters=100, pcg_tol=1e-5):

        self.C = C

        # armijo paramters
        self.armijo_alpha = armijo_alpha
        self.max_armijo_iters = 10
        self.armijo_beta = .5
        self.armijo_c = .1

        # newton step params
        self.max_descent_iters = max_descent_iters
        self.descent_tol = descent_tol

        # pcg params
        self.pcg_tol = pcg_tol
        self.max_pcg_iter = max_pcg_iters

        self.tikhonov = tikhonov

    def fit(self, S, Q=None, C=None):
        if Q is not None:
            try:
                np.linalg.cholesky(Q)
            except ValueError:
                raise Exception('Q is not SPD')
        else:
            Q = np.eye(S.shape[1])

        if C is None:
            if self.C is None:
                raise Exception('No sparsity pattern is given')
            C = self.C

        for _i in range(self.max_descent_iters):
            inv_Q = np.linalg.inv(Q)
            inv_Q = (inv_Q + inv_Q.T) / 2.
            g = self.grad_f(inv_Q, S) * C
            d = self.projPCG(inv_Q, g, C)
            alpha = self.armijo_linesearch(Q, S, g, d)
            if alpha is None:
                break

            np.multiply(alpha, d, out=d)
            np.add(Q, d, out=Q)
            Q = (Q + Q.T) / 2.

        return Q

    # target function : F(Q ; S) = - log(det(Q)) + tr(QS) + Tikhonov regularization
    def f(self, Q, S):
        try:
            L = np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            return None

        # log det(Q)
        log_diagonal = np.log(L.diagonal())
        log_det_Q = 2 * np.sum(log_diagonal)

        # trace(QS)
        tr_QS = (Q * S).sum()

        # tikhonov
        tikhonov_reg = self.tikhonov * np.trace(Q)

        return -log_det_Q + tr_QS + tikhonov_reg

    def grad_f(self, inv_q, S):
        return S - inv_q + self.tikhonov * np.eye(inv_q.shape[0])

    def armijo_linesearch(self, Q, S, g, d):
        alpha = self.armijo_alpha

        base_f_val = self.f(Q, S)
        d_dot_g = np.multiply(g, d).sum()
        for i in range(self.max_armijo_iters):
            Q_temp = Q + alpha * d

            curr_f_val = self.f(Q_temp, S)  # will throw exception if Q_temp is not SPD

            if curr_f_val is not None and curr_f_val <= base_f_val + alpha * self.armijo_c * d_dot_g:
                return alpha

            alpha = alpha * self.armijo_beta
        
        return None

    def projPCG(self, inv_Q, g, C):  # solving kron(inv_Q,inv_Q)d = -g
        b = -1 * g

        # Preconditioner
        M = np.kron(np.diag(inv_Q), np.diag(inv_Q)).reshape(inv_Q.shape)
        M += (1 - np.eye(inv_Q.shape[0])) * (inv_Q ** 2)
        M_inv = 1. / M

        x_k = -1 * g
        r_k = b - inv_Q.dot(x_k.dot(inv_Q)) * C
        z_k = r_k * M_inv
        r_0_r_0 = (r_k * r_k).sum()

        p_k = z_k.copy()
        p_k = (p_k + p_k.T) / 2.
        z_k_r_k = (z_k * r_k).sum()

        for k in range(self.max_pcg_iter):

            Ap_k = np.linalg.multi_dot([inv_Q, p_k, inv_Q]) * C

            denom = (p_k * Ap_k).sum()

            if denom < 1e-8:  # avoids small denominator error
                break

            # alpha_k
            alpha_k = (r_k * z_k).sum() / denom

            # update x_k
            x_k += alpha_k * p_k

            # update r_k, z_k
            r_k -= alpha_k * Ap_k
            np.multiply(r_k, M_inv, out=z_k)

            z_k_r_k_prev = z_k_r_k
            z_k_r_k = (z_k * r_k).sum()
            r_k_r_k = (r_k * r_k).sum()
            if np.sqrt(r_k_r_k/r_0_r_0) < self.pcg_tol:
                break
            beta_k = z_k_r_k / z_k_r_k_prev

            p_k *= beta_k
            p_k += z_k
            p_k = (p_k + p_k.T) / 2.

        return x_k
