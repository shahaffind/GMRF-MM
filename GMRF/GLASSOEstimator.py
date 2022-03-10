from scipy import linalg
import numpy as np


class GLASSOEstimator:

    def __init__(self, lambd, max_iters=10, alpha=1, tikhonov=0.0):
        self.alpha = alpha
        self.tikhonov = tikhonov
        self.lambd = lambd
        self.max_iters = max_iters

    def fit(self, S, Q=None):

        if Q is not None:
            try:
                linalg.cholesky(Q, lower=True)
            except ValueError:
                raise Exception('Q is not SPD')
        else:
            Q = np.eye(S.shape[1])

        for i in range(self.max_iters):
            g = self.grad_f(Q, S)
            d = self.NLCG(Q, g)
            if d is None:
                break
            alpha = self.armijo_linesearch_F(Q, S, g, d)
            if alpha is None:
                break

            np.add(Q, alpha * d, out=Q)
            Q = (Q + Q.T) / 2.
            # print(self.F(Q))

        Q[np.abs(Q) < 1e-8] = 0
        return Q

    # target function : F(Q ; S) = - log(det(Q)) + tr(QS) + L1
    def F(self, Q, S):
        try:
            L = linalg.cholesky(Q, lower=True)
        except:
            raise Exception('Q is not SPD')

        # det(Q) = det(L'L) = det(L)^2
        # log(det(Q)) = log(det(L)^2) = 2*log(det(L)) = 2*sum_i(log(L_ii))
        log_diagonal = np.log(L.diagonal())
        log_det_Q = 2 * np.sum(log_diagonal)

        tr_QS = np.multiply(Q, S).sum()

        tik = self.tikhonov * np.trace(Q)

        l1_reg = self.lambd * np.abs(Q - np.diag(np.diag(Q))).sum()

        return -log_det_Q + tr_QS + tik + l1_reg

    # grad of target function
    def grad_f(self, Q, S):
        inv_q = linalg.inv(Q)
        return S - inv_q + self.tikhonov * np.eye(Q.shape[0])

    def armijo_linesearch_F(self, Q, S, g, d):
        # params
        alpha = self.alpha
        beta = 0.5
        c = 0.1
        max_iter = 20

        base_f_val = self.F(Q, S)
        d_dot_g = np.multiply(g, d).sum()
        for i in range(max_iter):
            Q_temp = Q + alpha * d
            try:
                curr_f_val = self.F(Q_temp, S)  # will throw exception if Q_temp is not SPD
                if curr_f_val <= base_f_val + alpha * c * d_dot_g:
                    return alpha
            except:
                pass  # f throws when Q is not SPD
            alpha = alpha * beta

        return None

    def armijo_linesearch(self, x, d, sig, b, Q):
        # params
        alpha = 1.
        beta = 0.3
        max_iter = 10

        prev_f_val = self.target_nlcg(x + alpha * d, sig, b, Q)
        for i in range(max_iter):
            alpha = alpha * beta
            curr_f_val = self.target_nlcg(x + alpha * d, sig, b, Q)
            if curr_f_val > prev_f_val:
                return alpha / beta
            prev_f_val = curr_f_val
        # return alpha
        if prev_f_val < self.target_nlcg(x, sig, b, Q):
            return alpha
        return None

    def s_lambda(self, X, precon):
        X_diag = np.diag(np.diag(X))
        X_rest = X - X_diag
        sign = np.sign(X_rest)
        return sign * np.maximum(np.abs(X_rest) - precon * self.lambd, 0) + X_diag

    def target_nlcg(self, x, sig, b, Q):
        Qx_diag = np.diag(np.diag(Q + x))
        Qx_rest = Q + x - Qx_diag
        sig_x = sig.dot(x)
        return .5 * (sig_x * sig_x).sum() - (b * x).sum() + self.lambd * np.abs(Qx_rest).sum()

    def NLCG(self, Q, g):
        inv_Q = np.linalg.inv(Q)
        inv_Q = (inv_Q + inv_Q.T) / 2.
        b = -1 * g

        M = np.kron(np.diag(inv_Q), np.diag(inv_Q)).reshape(inv_Q.shape)
        M += (1 - np.eye(Q.shape[0])) * (inv_Q ** 2)
        M_inv = 1. / M
        np.fill_diagonal(M_inv, 1)

        # init + first iter
        x_k = np.zeros(g.shape)
        r_k = b - inv_Q.dot(x_k.dot(inv_Q))
        z_k = self.s_lambda(Q + M_inv * r_k, M_inv) - Q

        z_1_l1 = np.abs(z_k).sum()
        alpha_k = self.armijo_linesearch(x_k, z_k, inv_Q, b, Q)
        if alpha_k is None:
            return None
        x_k += alpha_k * z_k

        for k in range(min(Q.shape[0] - 1, 100)):
            z_k_prev = z_k

            r_k = b - np.linalg.multi_dot([inv_Q, x_k, inv_Q])
            z_k = self.s_lambda((Q + x_k) + M_inv * r_k, M_inv) - (Q + x_k)

            beta_pr = max(0, (z_k * (z_k - z_k_prev)).sum() / (z_k_prev * z_k_prev).sum())
            z_k += z_k_prev * beta_pr

            z_k_l1 = np.abs(z_k).sum()
            if z_k_l1 / z_1_l1 < 1e-3:
                break

            alpha_k = self.armijo_linesearch(x_k, z_k, inv_Q, b, Q)
            if alpha_k is None:
                break

            x_k += alpha_k * z_k

        return x_k
