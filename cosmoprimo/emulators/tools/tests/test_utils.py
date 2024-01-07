from cosmoprimo.emulators.tools import utils


def test_utils():

    import jax

    @jax.jit
    def fun(x):
        return utils.evaluate('a = x**2; b = 2; a + b', locals={'x': x})

    print(jax.grad(fun)(1.))


if __name__ == '__main__':

    test_utils()