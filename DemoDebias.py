import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import graphical_lasso

from Util.Sample import sample_by_precision
from Util import AnisotropicFilter
from GMRF.KnownSparsityEstimator import KnownSparsityEstimator as known_esimator
from GMRF.GLASSOEstimator import GLASSOEstimator as glasso_estimator
from GMRF.DebiasedEstimator import DebiasedEstimator as debiased_estimator


np.random.seed(0)

d = 20
n = 300

lapl = AnisotropicFilter.laplacian(d, d)

X = sample_by_precision(np.zeros(d**2), lapl, n)
emp_cov = (X.T.dot(X) * 1/n) + np.eye(d ** 2) * 1e-5

Vars = np.diag(emp_cov)
Vars_sqrt = np.sqrt(Vars)
V = np.diag(Vars_sqrt)
V_inv = np.diag(Vars_sqrt ** -1)  # each entry of V is the variance of the X_i
S = V_inv.dot(emp_cov).dot(V_inv)

w_true, _ = np.linalg.eigh(lapl)

plt.plot(w_true, label='True')


C = np.abs(lapl) > 1e-8
_known_estimator = known_esimator(max_descent_iters=10)
Q_known_normed = _known_estimator.fit(S, np.eye(d ** 2), C)
Q_known = V_inv.dot(Q_known_normed).dot(V_inv)
w_known, _ = np.linalg.eigh(Q_known)

plt.plot(w_known, label='Known Sparsity')

_glasso_estimator = glasso_estimator(lambd=.25, max_iters=10)
Q_glasso_normed = _glasso_estimator.fit(S, np.eye(d ** 2))
Q_glasso = V_inv.dot(Q_glasso_normed).dot(V_inv)
w_glasso, _ = np.linalg.eigh(Q_glasso)

plt.plot(w_glasso, label='GLASSO')

_debiased_estimator = debiased_estimator(_glasso_estimator)
Q_debiased_normed = _debiased_estimator.fit(S, Q_glasso_normed)
Q_debiased = V_inv.dot(Q_debiased_normed).dot(V_inv)
w_debiased, _ = np.linalg.eigh(Q_debiased)

plt.plot(w_debiased, label='Debiased')


plt.legend()
plt.show()
