from cosmoprimo.emulators.tools import utils


def test_utils():

    import jax

    @jax.jit
    def fun(x):
        return utils.evaluate('a = x**2; b = 2; a + b', locals={'x': x})

    print(jax.grad(fun)(1.))


    from jax import numpy as np
    from cosmoprimo.jax import exception

    def warn(z1, z2):
        if np.any(z2 < z1):
            import warnings
            warnings.warn(f"Second redshift(s) z2 ({z2}) is less than first redshift(s) z1 ({z1}).")

    def fun(z1, z2):
        exception(warn, z1, z2)

    z1 = np.linspace(0., 1., 10)
    z2 = np.linspace(0., 1., 10) - 1.

    fun = jax.jit(fun)
    fun(z1, z2)


def test_romberg():

    def function(x):
        return x**6

    from cosmoprimo.emulators.tools.jax import romberg
    from scipy.integrate import romberg as romberg_ref

    print(romberg(function, 0., 1.))
    print(romberg_ref(function, 0., 1.))


if __name__ == '__main__':

    test_utils()
    #test_romberg()