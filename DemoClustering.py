import numpy as np
from Util.AnisotropicFilter import create_D
from Util.Sample import sample_by_precision
from MixtureModel.GMM import GMM

from GMRF.KnownSparsityEstimator import KnownSparsityEstimator as known_esimator
from GMRF.GLASSOEstimator import GLASSOEstimator as glasso_estimator
from GMRF.DebiasedEstimator import DebiasedEstimator as debiased_estimator

from sklearn.metrics import normalized_mutual_info_score as nmi
from Util.VI import vi


def create_dataset(k_components, D, sample_range):
    # Random components
    sigmas = np.random.rand(k_components, D.shape[0])
    Qs = [D.T.dot(np.diag(s)).dot(D) for s in sigmas]
    mu = np.zeros(Qs[0].shape[0])

    Ns = np.random.randint(sample_range[0], sample_range[1], k_components)

    Xs = [sample_by_precision(mu, Qs[i], Ns[i]) for i in range(k_components)]
    X = np.vstack(Xs)
    y = np.concatenate([np.ones(Ns[i]) * i for i in range(k_components)])

    perm = np.random.permutation(y.shape[0])

    return X[perm], y[perm]


def run_test(dim, k_components, iters, range_per_component):

    D = create_D(dim, dim)

    C = np.abs(D.T.dot(D)) > 1e-8

    nmi_res = np.zeros(iters)
    vi_res = np.zeros(iters)

    for _i in range(iters):
        print(f'iter: {_i}')

        X, y = create_dataset(k_components, D, range_per_component)

        gmm = GMM(k_components, use_means=False, warm_start=True)
        # gmm = GMM(k_components, use_means=False, warm_start=False)

        # estimator = None
        # estimator = glasso_estimator(lambd=0.3, max_iters=15)
        # estimator = known_esimator(C=C, max_descent_iters=15)
        estimator = debiased_estimator(base_estimator=glasso_estimator(lambd=0.3, max_iters=10))

        gmm.set_Q_estimator(estimator)
        gmm.set_dirichlet_prior(0.001 * X.shape[0] / k_components)
        gmm.fit(X, max_iters=15, verbose=True)

        y_predict = np.argmax(gmm.get_proba(X), axis=1)

        nmi_res[_i] = nmi(y, y_predict, average_method='arithmetic')
        vi_res[_i] = vi(k_components, y, k_components, y_predict)
        print(f'\t\tnmi: {nmi_res[_i]}, vi: {vi_res[_i]}')

    print(f'##########')
    print(f'nmi: mean={nmi_res.mean()}, std={nmi_res.std()}')
    print(f'vi: mean={vi_res.mean()}, std={vi_res.std()}')
    print(f'##########')


if __name__ == "__main__":
    run_test(dim=10, k_components=5, iters=10,
             range_per_component=(1500, 2500))
