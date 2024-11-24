import numpy as np
from scipy import optimize

from cosmoprimo.utils import LeastSquareSolver, DistanceToRedshift


def test_least_squares():

    for compute_inverse in [False, True]:

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

            lss = LeastSquareSolver(gradient, precision, compute_inverse=compute_inverse)
            result = lss(y)
            assert np.allclose(result, result_ref, rtol=1e-2, atol=1e-2)

            lss_c = LeastSquareSolver(gradient, precision, constraint_gradient=np.ones((len(gradient), 1)), compute_inverse=compute_inverse)
            constraint = 0.42
            result = lss_c(y, constraint=constraint)
            assert lss_c.chi2() >= lss.chi2()
            assert np.allclose(sum(result), constraint)

            weights = np.arange(len(gradient))
            lss_c = LeastSquareSolver(gradient, precision, constraint_gradient=np.column_stack([np.ones(len(gradient)), weights]), compute_inverse=compute_inverse)
            constraint = [0.42, 2.]
            result = lss_c(y, constraint=constraint)
            assert lss_c.chi2() >= lss.chi2()
            assert np.allclose(sum(result), constraint[0])
            assert np.allclose(sum(r * w for r, w in zip(result, weights)), constraint[1])

        result_ref = LeastSquareSolver(gradient, precision=np.eye(x.size), compute_inverse=compute_inverse)(y)
        for precision in [1., np.ones_like(x)]:
            result = LeastSquareSolver(gradient, precision=precision, compute_inverse=compute_inverse)(y)
            assert np.allclose(result, result_ref)

        lss = LeastSquareSolver(gradient, precision=np.eye(x.size), compute_inverse=compute_inverse)
        result_ref = lss(y)
        ys = np.array([y] * 12)
        result = lss(ys)
        assert result.shape == (len(ys), len(gradient))
        assert np.allclose(result, result_ref)
        assert lss.model().shape == ys.shape
        assert lss.chi2().shape == (len(ys), )

        gradient = np.ones_like(x)
        lss = LeastSquareSolver(gradient, precision=np.eye(x.size), compute_inverse=compute_inverse)
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


def test_jax():
    import jax
    from jax import numpy as jnp
    from cosmoprimo.jax import romberg, odeint, bisect

    def fun(x, a=0.):
        return x**3 + a

    limits = jnp.array([-1., 1.])
    for atol in [1e-3, 1e-6]:
        assert np.allclose(bisect(fun, *limits, xtol=atol), 0., atol=atol)

    def fun(x, a=0.):
        return x**3 - a

    print(jax.jacfwd(lambda a: bisect(lambda x: fun(x, a=a), *limits, xtol=atol))(0.1))

    def fun(x):
        return x

    assert jnp.allclose(romberg(fun, 0., 1.), 1. / 2.)
    assert jnp.allclose(romberg(fun, jnp.array(0.), jnp.array(1.)), 1. / 2.)

    def fun(x):
        toret = jnp.column_stack([x, x**2]).reshape(x.shape + (2,))
        return toret

    assert jnp.allclose(romberg(fun, jnp.array(0.), jnp.array(1.)), jnp.array([1. / 2., 1. / 3.]))

    def integrand(y, z):
        return z

    print(odeint(integrand, 0., jnp.linspace(0., 1., 100)))




if __name__ == '__main__':

    test_least_squares()
    test_redshift_array()
    test_jax()
