import numpy as np
from scipy import optimize

from cosmoprimo.utils import SolveLeastSquares, DistanceToRedshift


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

        sls = SolveLeastSquares(gradient, precision)
        result = sls(y)
        assert np.allclose(result, result_ref, rtol=1e-2, atol=1e-2)

    result_ref = SolveLeastSquares(gradient, precision=np.eye(x.size))(y)
    for precision in [1., np.ones_like(x)]:
        result = SolveLeastSquares(gradient, precision=precision)(y)
        assert np.allclose(result, result_ref)

    sls = SolveLeastSquares(gradient, precision=np.eye(x.size))
    result_ref = sls(y)
    ys = np.array([y] * 12)
    result = sls(ys)
    assert result.shape == (len(ys), len(gradient))
    assert np.allclose(result, result_ref)
    assert sls.model().shape == ys.shape
    assert sls.chi2().shape == (len(ys), )

    gradient = np.ones_like(x)
    sls = SolveLeastSquares(gradient, precision=np.eye(x.size))
    assert sls(y).ndim == 0
    assert sls(ys).shape == (len(ys), )


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
    test_redshift_array()
