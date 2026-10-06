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

            solver = LeastSquareSolver(gradient, precision, compute_inverse=compute_inverse)
            result = solver(y)
            assert np.allclose(result, result_ref, rtol=1e-2, atol=1e-2)

            lss_c = LeastSquareSolver(gradient, precision, constraint_gradient=np.ones((len(gradient), 1)), compute_inverse=compute_inverse)
            constraint = 0.42
            result = lss_c(y, constraint=constraint)
            assert lss_c.chi2() >= solver.chi2()
            assert np.allclose(sum(result), constraint)

            weights = np.arange(len(gradient))
            lss_c = LeastSquareSolver(gradient, precision, constraint_gradient=np.column_stack([np.ones(len(gradient)), weights]), compute_inverse=compute_inverse)
            constraint = [0.42, 2.]
            result = lss_c(y, constraint=constraint)
            assert lss_c.chi2() >= solver.chi2()
            assert np.allclose(sum(result), constraint[0])
            assert np.allclose(sum(r * w for r, w in zip(result, weights)), constraint[1])

        result_ref = LeastSquareSolver(gradient, precision=np.eye(x.size), compute_inverse=compute_inverse)(y)
        for precision in [1., np.ones_like(x)]:
            result = LeastSquareSolver(gradient, precision=precision, compute_inverse=compute_inverse)(y)
            assert np.allclose(result, result_ref)

        solver = LeastSquareSolver(gradient, precision=np.eye(x.size), compute_inverse=compute_inverse)
        result_ref = solver(y)
        ys = np.array([y] * 12)
        result = solver(ys)
        assert result.shape == (len(ys), len(gradient))
        assert np.allclose(result, result_ref)
        assert solver.model().shape == ys.shape
        assert solver.chi2().shape == (len(ys), )

        gradient = np.ones_like(x)
        solver = LeastSquareSolver(gradient, precision=np.eye(x.size), compute_inverse=compute_inverse)
        assert solver(y).ndim == 0
        assert solver(ys).shape == (len(ys), )

        def test(factor):
            gradient = np.ones_like(x)
            return LeastSquareSolver(gradient, precision=factor * np.eye(x.size), compute_inverse=compute_inverse)

        import jax
        jax.jit(test)(1.)


def test_redshift_array():

    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    zmax = 10.
    distance = cosmo.comoving_radial_distance
    redshift = DistanceToRedshift(distance=distance, zmax=zmax, nz=4096)
    z = np.random.uniform(0., 2., 10000)
    assert np.allclose(redshift(distance(z)), z, atol=1e-6)

    def test(params):
        cosmo = DESI(**params, engine='bbks')
        return DistanceToRedshift(distance=cosmo.comoving_radial_distance, zmax=10.)

    import jax
    jax.jit(test)(dict(h=0.7))(100.)
    jax.jacfwd(lambda params: test(params)(100.))(dict(h=0.7))


def test_jax():
    import jax
    from jax import numpy as jnp
    from cosmoprimo.jax import romberg, odeint, bisect, Interpolator1D, Interpolator2D, Partial

    def test(factor):

        def fun(factor, x):
            return x * factor

        return Partial(fun, factor)

    def test(factor):
        x = jnp.linspace(0., 1., 100)
        fun = factor * jnp.linspace(0., 1., 100)
        toret1 = Interpolator1D(x, fun)
        y = jnp.linspace(0., 1., 10)
        fun = factor * jnp.linspace(0., 1., 1000).reshape(100, 10)
        toret2 = Interpolator2D(x, y, fun)
        return toret1, toret2

    print(jax.jit(test)(1.))

    def fun(x, a=0.):
        return x**3 + a

    limits = jnp.array([-1.02, 2.01])
    for atol in [1e-3, 1e-6]:
        tmp = bisect(fun, limits, xtol=atol)
        assert np.allclose(tmp, 0., atol=atol)

    def fun(x, a=0.):
        return x**3 - a

    print(jax.jacrev(lambda a: bisect(lambda x: fun(x, a=a), limits, xtol=atol))(1.))

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




def test_bisect_numpy():
    """Non-JAX bisect: both methods, and the rounding regime that used to break ridders."""
    from cosmoprimo.jax import bisect

    for method in ['ridders', 'bisection']:
        # 'bisection' used to raise UnboundLocalError: `sign` was assigned in the ridders branch,
        # which made it a local of the enclosing function everywhere.
        assert np.allclose(bisect(lambda x: x**2 - 4., [0., 5.], xtol=1e-10, method=method), 2., atol=1e-6), method

    # theta_MC_100 ~ 1.04 has a ulp of 2.2e-16, so theta_MC_100 - target is quantised near the
    # root and can round to the wrong side of zero. Ridders then finds no sign change among low,
    # mid, new and high, and cannot shrink the bracket any further: it used to spin to maxiter and
    # fall off the end of the loop returning None, which Cosmology.solve fed straight back into a
    # clone as h=None ('unsupported operand type(s) for **: NoneType and int'). Observed with CAMB
    # while solving h for theta_MC_100, and with CLASS on other nodes.
    root, target = 0.6533955615074739, 1.0408640431821163

    def residual(h):
        return (target + 0.336 * (h - root)) - target

    for maxiter in [40, 60]:
        value = bisect(residual, [0.627865, 0.6536818], xtol=1e-9, maxiter=maxiter)
        assert value is not None, 'ridders returned None on a stalled bracket'
        assert np.allclose(value, root, atol=1e-12), value

    # Genuine non-convergence must be an error, never a silent None.
    try:
        bisect(lambda x: x - 0.4321, [0., 1.], xtol=1e-15, maxiter=1)
    except ValueError:
        pass
    else:
        raise AssertionError('non-convergence did not raise')

if __name__ == '__main__':

    test_jax()
    test_bisect_numpy()
    test_least_squares()
    test_redshift_array()
