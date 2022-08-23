import numpy as np
from scipy import optimize

from cosmoprimo.utils import LeastSquareSolver, DistanceToRedshift


def test_least_squares():

    x = np.linspace(1, 100, 10)
    gradient = np.array([1. / x, np.ones_like(x), x, x ** 2, x ** 3])

    covs = [np.diag(x), np.diag(x) + 0.1]
    rng = np.random.RandomState(seed=42)
    y = rng.uniform(0., 1., x.size)

    for cov in covs:

        precision = np.linalg.inv(cov)

        def chi2(pars):
            delta = y - pars.dot(gradient)
            return np.sum(delta.dot(precision).dot(delta.T))

        x0 = np.zeros(len(gradient))
        result_ref = optimize.minimize(chi2, x0=x0, args=(), method='Nelder-Mead', tol=1e-6, options={'maxiter': 1000000}).x

        lss = LeastSquareSolver(gradient, precision)
        result = lss(y)
        assert np.allclose(result, result_ref, rtol=1e-2, atol=1e-2)

        lss_c = LeastSquareSolver(gradient, precision, constraint_gradient=np.ones((len(gradient), 1)))
        constraint = 0.42
        result = lss_c(y, constraint=constraint)
        assert lss_c.chi2() >= lss.chi2()
        assert np.allclose(sum(result), constraint)

        weights = np.arange(len(gradient))
        lss_c = LeastSquareSolver(gradient, precision, constraint_gradient=np.array([np.ones(len(gradient)), weights]).T)
        constraint = [0.42, 2.]
        result = lss_c(y, constraint=constraint)
        assert lss_c.chi2() >= lss.chi2()
        assert np.allclose(sum(result), constraint[0])
        assert np.allclose(sum(r * w for r, w in zip(result, weights)), constraint[1])

    result_ref = LeastSquareSolver(gradient, precision=np.eye(x.size))(y)
    for precision in [1., np.ones_like(x)]:
        result = LeastSquareSolver(gradient, precision=precision)(y)
        assert np.allclose(result, result_ref)

    lss = LeastSquareSolver(gradient, precision=np.eye(x.size))
    result_ref = lss(y)
    ys = np.array([y] * 12)
    result = lss(ys)
    assert result.shape == (len(ys), len(gradient))
    assert np.allclose(result, result_ref)
    assert lss.model().shape == ys.shape
    assert lss.chi2().shape == (len(ys), )

    gradient = np.ones_like(x)
    lss = LeastSquareSolver(gradient, precision=np.eye(x.size))
    assert lss(y).ndim == 0
    assert lss(ys).shape == (len(ys), )


def test_redshift_array():

    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    zmax = 10.
    distance = cosmo.comoving_radial_distance
    redshift = DistanceToRedshift(distance=distance, zmax=zmax, nz=4096)
    z = np.random.uniform(0., 2., 10000)
    assert np.allclose(redshift(distance(z)), z, atol=1e-6)


if __name__ == '__main__':

    test_least_squares()
    #test_redshift_array()
