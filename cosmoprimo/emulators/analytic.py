r"""The analytic formulae the cosmology emulators divide out, traceable.

One home for what :mod:`cosmoprimo.emulators.cosmology` and desilike's
``theories.primordial_cosmology`` both need: the amplitude spellings and the per-leg optical-depth
screening of a :math:`C_\ell`, the analytic w0waCDM background scalars, the analytic
:math:`\theta_\mathrm{MC}` and its inverse, and the Eisenstein & Hu drag-epoch scales. None of
these is a truth -- each is a preconditioner: a smooth function of the same parameters with the
right scalings, divided out before a fit and multiplied back at prediction, so what the
interpolant sees is the Boltzmann code's correction to it. What a formula gets wrong stays on the
grid.

Every function here runs under a jax trace when given traced inputs (``jax.numpy`` through
:mod:`cosmoprimo.jax`, plain numpy otherwise) and never casts to float: desilike divides them out
inside a jitted prediction.
"""

import numpy as np

from cosmoprimo import constants
from cosmoprimo.cosmology import Cosmology
from cosmoprimo.jax import jax, jit, numpy as jnp, numpy_jax


#: :math:`A_s` in every spelling cosmoprimo accepts; an emulator should see only one of them.
#: Read off `Cosmology`'s own alias table rather than listed here: a spelling added there would
#: otherwise be silently missed, and take the amplitude onto the grid instead of scaling it --
#: an emulator that is quietly worse, not one that fails.
AMPLITUDES = ('A_s', 'logA') + tuple(Cosmology._alias_parameters['logA'])


def amplitude(params):
    """:math:`A_s` in whatever spelling was given, or None if the space varies no amplitude.

    The spellings come off `Cosmology`'s own alias table rather than a list here, so one added
    there is not silently missed -- which would put the amplitude on the grid instead of scaling
    it: quietly worse, not failing.
    """
    for name in AMPLITUDES:
        if name in params:
            return params[name] if name == 'A_s' else 1e-10 * jnp.exp(params[name])
    return None


def harmonic_scaling(names, amplitude=None, tau=None):
    r"""``{name: factor}`` to divide out of the :math:`C_\ell` named in *names*.

    The amplitude, :math:`C_\ell \propto A_s`, and one :math:`e^{-\tau}` per screened leg: ``tt``
    and ``ee`` carry :math:`e^{-2\tau}`, ``tp`` and ``ep`` one factor, ``pp`` none. Both are
    flattenings rather than removals once lensing is applied and below :math:`\ell \sim 30`,
    where reionization puts power back. A name's spectrum is its last dotted component
    (``'lensed_cl.tt'``, ``'harmonic.lensed_cl|ellmax=2500.tt'``), and the legs are counted from
    it rather than listed: getting the per-leg count wrong is a silent factor of :math:`e^{\tau}`.
    """
    factors = {}
    for name in names:
        spectrum = name.rsplit('.', 1)[-1]
        factor = 1. if amplitude is None else amplitude
        if tau is not None:
            factor = factor * numpy_jax(tau).exp(-tau * sum(leg != 'p' for leg in spectrum))
        factors[name] = factor
    return factors


def nonzero(values):
    """``values`` -- an array or a scalar -- with any exact zero replaced by one.

    Applied to every analytic divisor before it is handed to :meth:`~SectionEmulator.scaling`.
    The one place it is not merely defensive is the background: the default grid ends at z = 0,
    where ``comoving_radial_distance`` is exactly zero, and 0/0 would put a NaN straight into the
    training data.

    The substitution never shows: ``transform`` and ``inverse_transform`` both read the factors
    from ``scaling``, so whatever is divided out is multiplied back, one for one.
    """
    xnp = numpy_jax(values)
    values = xnp.asarray(values)
    return xnp.where(values == 0., 1., values)


def fourier_analytic_scales(z, cosmo, nsteps=200, nodes=48):
    r"""Baseline background scalars ``(invE, DM, D, f)`` at a single redshift.

    Named for what they are for: the growth carries a Fourier spectrum's redshift dependence,
    one factor of :math:`D` per density leg and a further :math:`f` per velocity one, and it is
    divided out with them. ``invE`` and ``DM`` come along because they are the same integration,
    and the background sector divides those out in turn.

    Evaluated at the one redshift asked for, nothing tabulated: cosmoprimo's
    ``DefaultBackground`` would build two C2 cubic splines (banded solves over 201 and 119 knots)
    and read them at a single z, rebuilt on every call because the caller clones a fresh
    cosmology each time -- measured at 81% of this function and 15x its total cost.

    Three choices worth knowing:

    * the growth is early-time normalised, D ~ a in matter domination.
      Normalising by D(0) makes D a relative growth, and the c_D / c_DM corrections then have to
      absorb the cosmology dependence of D(0) itself: their spread over the w0-wa box goes from
      ~1e-3 to 12%.
    * ``DM`` is the transverse distance, matching what ``qper`` is built from, since the
      correction it anchors is ``qper / (analytic DM ratio)``.  Curvature enters as a series in
      :math:`x^2` for :math:`\sinh(x)/x`, analytic at ``Omega_k = 0`` and needing no branch.
    * massive neutrinos count as matter, :math:`(1 + z)^3`.  True at the redshifts used, not at
      the start of the growth integration (z ~ 3000, kT_nu ~ m), and that is what is left once
      the initial condition below is right: scored against CLASS it is a floor of ~1e-3 reached
      at m_ncdm = 0.45, where every initial condition gives the same answer.
      (The comparison against the tabulated implementation quoted here previously -- ``invE``,
      ``DM``, ``f`` to <1e-6 and ``D`` to 2.9e-3 -- was measured with the old ``D = D' = a`` at
      ``eta_start = -6`` and has not been re-run.)

    A preconditioner, not a truth, so a smooth shift is absorbed by the fitted correction.  What
    would not be absorbed is numerator and denominator coming from different functions, which is
    why :meth:`desilike.theories.galaxy_clustering.template.ScalingScalarsEmulator.set_ref_fiducial` recomputes its anchor rather than
    trusting the stored one.

    Parameters
    ----------
    z : float
        Redshift.  Must be concrete: it sets the (static) integration grids.
    cosmo : Cosmology
        Read for its z = 0 density parameters only; no background is constructed.
    nsteps : int, default=200
        RK4 steps from ``ln a = -6`` to ``ln a = -ln(1 + z)``.
    nodes : int, default=48
        Gauss-Legendre nodes for the comoving distance.

        Both defaults are converged: 200 -> 800 and 48 -> 96 leave every scalar unchanged to the
        digit, so raising them only costs time (15x down to 6.8x).
    """
    if np.ndim(z) != 0:
        raise ValueError('z must be a scalar, got shape {}'.format(np.shape(z)))

    def total(name):
        return jnp.sum(jnp.atleast_1d(jnp.asarray(cosmo[name])))

    omega_cb = total('Omega_cdm') + total('Omega_b')
    omega_m = omega_cb + total('Omega_ncdm')          # as matter, see above
    omega_r = total('Omega_g') + total('Omega_ur')
    omega_k, omega_de = total('Omega_k'), total('Omega_de')
    w0, wa = total('w0_fld'), total('wa_fld')

    def densities(redshift):
        """Unnormalised sum(rho_i / rho_crit0), and its dark-energy part, at *redshift*."""
        opz = 1. + redshift
        dark = opz**(3. * (1. + w0 + wa)) * jnp.exp(-3. * wa * redshift / opz)
        return omega_m * opz**3 + omega_r * opz**4 + omega_k * opz**2 + omega_de * dark, dark

    # E(0) = 1 exactly, as cosmoprimo's is.  The sum at z = 0 is 1.4e-7 off unity here, because
    # Omega_ncdm today is not exactly its matter-equivalent.
    norm = densities(0.)[0]

    def efunc2(redshift):
        return densities(redshift)[0] / norm

    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(nodes)
    hubble_distance = constants.c / 1e3 / 100.
    chi = hubble_distance * jnp.sum(jnp.asarray(0.5 * z * gauss_weights)
                                    / jnp.sqrt(efunc2(jnp.asarray(0.5 * z * (gauss_nodes + 1.)))))
    curvature = omega_k * (chi / hubble_distance)**2
    distance = chi * (1. + curvature / 6. + curvature**2 / 120. + curvature**3 / 5040.)

    # The growth equation in eta = ln a, exactly as cosmoprimo writes it: y = (D, D') with
    # y' = A y, A = [[0, 1], [3/2 Omega_cb, -2 - dlnH/dlna]], stepped with RK4.
    # ln(a) grid.  It ends at z, so there is no run to z = 0 and nothing to interpolate back.
    #
    # It does not start where cosmoprimo's does, and the initial condition is not cosmoprimo's
    # `D = D' = a` either -- see the initial condition below.  Deliberate: that pair is the
    # growing mode only where radiation and dark energy are both negligible, and at a = e^-6
    # neither is.  Measured 2026-09-04 against CLASS, over 9 (h, omega_cdm) cells x 10 CPL
    # points, on the quantity the routing cannot absorb (how much D_analytic / D_true moves with
    # (w0, wa) at fixed everything else, since ScalingScalarsEmulator does not expand them):
    #
    #   eta_start   initial condition                median      max
    #      -6       D = D' = a  (cosmoprimo's)      7.75e-03   1.42e-02
    #      -8       D = D' = a                      8.78e-04   1.46e-03
    #     -10       Meszaros amplitude, D' = a      6.24e-04   1.23e-03
    #      -8       Meszaros amplitude, local D'    1.74e-04   3.20e-04   <- this one
    eta_start = -8.
    eta = np.linspace(eta_start, -np.log1p(z), nsteps + 1)
    eta_prev, eta_next = eta[:-1], eta[1:]
    step = jnp.asarray(eta_next - eta_prev)

    def coefficients(eta_nodes):
        redshift = np.exp(-eta_nodes) - 1.
        summed, dark = densities(redshift)
        opz = 1. + redshift
        w_fld = w0 + redshift / opz * wa
        # adotdot / (a H^2) = 1 + dlnH / dlna
        acceleration = -0.5 * (1. - omega_k * opz**2 / summed + omega_r * opz**4 / summed
                               + 3. * w_fld * omega_de * dark / summed)
        return -1. - acceleration, 1.5 * omega_cb * opz**3 / summed

    zeros, ones = jnp.zeros_like(step), jnp.ones_like(step)

    def amatrix(coefficient_1, coefficient_2):
        return jnp.stack([jnp.stack([zeros, ones], axis=-1),
                          jnp.stack([coefficient_2, coefficient_1], axis=-1)], axis=-2)

    identity = jnp.stack([jnp.stack([ones, zeros], axis=-1),
                          jnp.stack([zeros, ones], axis=-1)], axis=-2)
    amat_first = amatrix(*coefficients(eta_prev))
    amat_mid = amatrix(*coefficients(0.5 * (eta_prev + eta_next)))
    amat_last = amatrix(*coefficients(eta_next))
    scale = step[..., None, None]
    kmat1 = amat_first
    kmat2 = amat_mid @ (identity + scale / 2. * kmat1)
    kmat3 = amat_mid @ (identity + scale / 2. * kmat2)
    kmat4 = amat_last @ (identity + scale * kmat3)
    steps = identity + scale / 6. * (kmat1 + 2. * kmat2 + 2. * kmat3 + kmat4)

    # Only the endpoint is wanted, so reduce pairwise (log depth) rather than scanning: a
    # sequential scan over 200 steps is 200 tiny kernels, which is the cost this is avoiding.
    while steps.shape[0] > 1:
        if steps.shape[0] % 2:
            steps = jnp.concatenate([steps, identity[:1]], axis=0)
        steps = steps[1::2] @ steps[0::2]          # the later step multiplies on the left
    # Initial condition, growing mode at eta_start, in two pieces.
    #
    # Amplitude -- the Meszaros growing mode `D = 1 + 3y/2`, y = a / a_eq, normalised so that
    # `D -> a` once matter dominates: `D = a + (2/3) a_eq`.  That is the same znorm = 0
    # convention the c_D / c_DM corrections are built on, now imposed correctly rather than only
    # where radiation has already become negligible (rho_r / rho_m is still 0.10 at a = e^-6).
    #
    # Slope -- the growing root of the local indicial equation, `p^2 - c1 p - c2 = 0` for
    # `D'' = c2 D + c1 D'` (p = 1 in the EdS limit, where c1 = -1/2 and c2 = 3/2).  This is the
    # piece that matters here: `D' = D` injects a decaying mode whose size depends on how far
    # from EdS the start is, and dark energy is what moves that with (w0, wa) --
    # rho_DE / rho_m at a = e^-6 runs 4e-12 at w0 + wa = -1.5 but 2.0e-2 at -0.26.  The
    # amplitude piece alone buys nothing (7.7e-03 in the table above, i.e. unchanged): it is
    # w0/wa-independent and cancels in the ratio the corrections take.
    #
    # The residual is then the neutrinos-as-matter approximation, which no initial condition can
    # remove: at m_ncdm = 0.45 every variant converges to ~1e-03.  Below that the gain is 28x
    # (m_ncdm -> 0), 55x (0.06), 33x (0.25), and the optimal eta_start does not move with the
    # mass, so -8 is not cancelling against it.
    start = np.exp(eta_start)
    coefficient_1, coefficient_2 = coefficients(eta_start)
    index = 0.5 * (coefficient_1 + jnp.sqrt(coefficient_1**2 + 4. * coefficient_2))
    initial = start + 2. / 3. * omega_r / omega_m
    growth, growth_prime = steps[0] @ jnp.stack([initial, index * initial])

    return {'invE': 1. / jnp.sqrt(efunc2(z)), 'DM': distance,
            'D': growth, 'f': growth_prime / growth}


#: Speed of light, km/s.
_CLIGHT = 299792.458
#: <p>/T for a relativistic Fermi-Dirac gas: sets where the a^-4 -> a^-3 turn happens.
_FERMI_DIRAC_MOMENTUM = 3.15137
#: Boltzmann constant, eV/K.
_KELVIN_TO_EV = 8.617333262e-5


@jit(static_argnames=('na',))
def theta_analytic(h, omega_b, omega_cdm, w0=-1., wa=0., m_ncdm=(0.06,), N_ur=2.0328,
                            T_cmb=2.7255, T_ncdm_over_cmb=0.71611, na=2048):
    r"""``theta_cosmomc`` from closed-form background quantities alone.

    The same definition cosmoprimo derives -- :math:`r_s h / D_M(z_\star)` with the Hu & Sugiyama
    :math:`z_\star` -- but every ingredient analytic, so this jits, grads and **vmaps** and needs
    no :class:`Cosmology` instance.

    That is the point. Deriving ``theta_cosmomc`` the ordinary way runs the background, whose only
    non-closed-form piece is the massive-neutrino density: a Fermi-Dirac integral cosmoprimo caches
    in a spline per cosmology instance, and so rebuilds inside a trace at every call. Measured,
    that route costs 1.2 ms jitted and gets worse under ``vmap`` (1.86 ms/point); this is 0.21 ms
    jitted and 0.051 ms/point at ``vmap`` 256. It lives here rather than in cosmoprimo because it
    exists for :meth:`desilike.theories.primordial_cosmology.HarmonicEmulator.to_training`, which runs inside the trace where the exact value
    cannot be computed at all -- it is behind a ``pure_callback``.

    ``m_ncdm`` is an ordinary argument, not a captured constant, so it can be emulated later.

    Compiled, because a basis change is applied point by point rather than batched: a space is
    mapped into the training basis by transforming each of its (100000, by default) points, and
    ``Space.map`` has no batched entry point to hand them all over at once. Eager that is 3.9 ms
    a point and 6.4 minutes of construction; compiled, 0.126 ms and 13 seconds. Under ``vmap`` it
    would be 0.0074 ms, which is what to reach for if the map ever learns to batch.

    Accuracy against the exact derivation over a CMB-only w0waCDM box, h in [0.60, 0.85]: bias
    +1.6 sigma(theta), **scatter 0.075 sigma(theta)**. The bias is a near-constant offset and
    calibrates out; uncalibrated it is not ignorable, a box centred on a mis-converted theta having
    been measured 5.3 sigma off.

    Two traps, both silent:

    * do not renormalise :math:`\omega_\nu(a)` so that :math:`\omega_\nu(1) = \sum m_\nu / 93.14`.
      That rescales the whole profile and corrupts the relativistic limit, which
      :math:`\omega_\gamma` fixes: measured, 3.7% (158 sigma) of bias.
    * today's neutrino density must be evaluated at :math:`a = 1`, not taken as the last element of
      whatever grid is in hand -- on the :math:`r_s` grid that is the density at :math:`a_\star`,
      which drives :math:`\omega_{de}` negative and the result to NaN.
    """
    omega_g = 2.4735e-5 * (T_cmb / 2.7255) ** 4
    omega_ur = N_ur * 7. / 8. * (4. / 11.) ** (4. / 3.) * omega_g

    def omega_ncdm(a):
        masses = jnp.atleast_1d(jnp.asarray(m_ncdm))
        relativistic = 7. / 8. * (4. / 11.) ** (4. / 3.) * omega_g
        temperature = _KELVIN_TO_EV * T_cmb * T_ncdm_over_cmb
        y = masses[:, None] * a[None, :] / (_FERMI_DIRAC_MOMENTUM * temperature)
        return jnp.sum(relativistic / a[None, :] ** 4 * jnp.sqrt(1. + y ** 2), axis=0)

    omega_ncdm0 = omega_ncdm(jnp.ones(1))[0]
    omega_m = omega_b + omega_cdm + omega_ncdm0
    omega_de = h ** 2 - omega_m - omega_g - omega_ur

    def one_over_a2H(a):
        de = omega_de * a ** (-3. * (1. + w0 + wa)) * jnp.exp(-3. * wa * (1. - a))
        total = (omega_g + omega_ur) / a ** 4 + (omega_b + omega_cdm) / a ** 3 + omega_ncdm(a) + de
        return _CLIGHT / (a ** 2 * 100. * jnp.sqrt(total))

    zstar = 1048. * (1. + 0.00124 * omega_b ** -0.738) * (
        1. + (0.0783 * omega_b ** -0.238 / (1. + 39.5 * omega_b ** 0.763))
        * omega_m ** (0.560 / (1. + 21.1 * omega_b ** 1.81)))
    astar = 1. / (1. + zstar)

    a_rs = jnp.exp(jnp.linspace(jnp.log(1e-8), jnp.log(astar), na))
    sound_speed = (3. * (1. + 3e4 * a_rs * omega_b)) ** -0.5
    rs = jnp.trapezoid(one_over_a2H(a_rs) * sound_speed, a_rs)
    a_dm = jnp.exp(jnp.linspace(jnp.log(astar), 0., na))
    return rs / jnp.trapezoid(one_over_a2H(a_dm), a_dm)



@jit(static_argnames=('iterations', 'na'))
def solve_theta_analytic(target, omega_b, omega_cdm, limits=(0.2, 2.5), iterations=44, na=2048,
                          **kwargs):
    """The ``h`` whose analytic ``theta_MC_100`` is *target*, by bisection on a closed-form function.

    No engine is touched, so this works inside a trace, which is what
    :meth:`desilike.theories.primordial_cosmology.HarmonicEmulator.from_training` needs -- the exact solve runs a background per iteration and
    cannot be traced at all.

    Compiled for the reason :func:`theta_analytic` is, and one of its own: a bisection runs that
    closed form once per iteration, 44 of them, so an eager node evaluation would pay 44 times
    3.9 ms to place a single node.

    Deliberately not used to seed :meth:`cosmoprimo.Cosmology.solve`: tried and measured slower,
    1.23 s/solve against 0.80, because ``bracket`` and ridders still spend their ~10 exact
    background evaluations wherever they start.

    A bisection whose bracket does not contain the root returns ``nan``, not an endpoint. It is
    the one thing this has to get right: :meth:`desilike.theories.primordial_cosmology.HarmonicEmulator.from_training` calls it to turn a
    node's ``theta_MC_100`` into the ``h`` the calculator is evaluated at, so an endpoint is a
    node fitted at one theta and recorded at another, and a Chebyshev fit mixes that into every
    coefficient. Measured on the CMB w0waCDM box at nsigma 3.75, the old ``(0.2, 1.5)`` bracket
    railed 22 of 817 nodes at ``h = 1.5``, up to 0.0046 in ``theta_MC_100`` -- 19 sigma of the
    chain's own width -- and said nothing. ``nan`` instead surfaces as an unevaluable node, which
    the training reports and stops on.

    The default bracket is wide enough for the whitened box the CMB-only w0waCDM posterior asks
    for, whose corners reach ``h = 1.57``; ``iterations`` keeps the resolution it had.
    """
    def residual(h):
        return 100. * theta_analytic(h, omega_b, omega_cdm, na=na, **kwargs) - target

    low, high = limits
    flow, fhigh = residual(low), residual(high)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        cond = residual(middle) * flow > 0.
        low = jnp.where(cond, middle, low)
        high = jnp.where(cond, high, middle)
    return jnp.where(flow * fhigh > 0., jnp.nan, 0.5 * (low + high))


def eisenstein_hu_scales(cosmo):
    r"""The Eisenstein & Hu (1998) fitting formulae for :math:`z_\mathrm{drag}` and
    :math:`r_s(z_\mathrm{drag})`, in Mpc/h.

    Transcribed from ``EisensteinHuEngine._set_rsdrag`` rather than called through it, because
    that engine refuses massive neutrinos, curvature and dark energy. A preconditioner does not have
    to be correct physics, only a smooth function of the same parameters with roughly the right magnitude;
    what it gets wrong stays on the grid and is interpolated as before.
    """
    omega_m, omega_b = cosmo['omega_m'], cosmo['omega_b']
    # jax when the cosmology is traced (desilike's ThermodynamicsEmulator divides this out
    # inside a jitted prediction), numpy otherwise; no cast to float for the same reason
    xnp = numpy_jax(omega_m, omega_b)
    theta_cmb = cosmo['T_cmb'] / 2.7
    z_eq = 2.5e4 * omega_m * theta_cmb**(-4) - 1.
    k_eq = 0.0746 * omega_m * theta_cmb**(-2)
    b1 = 0.313 * omega_m**(-0.419) * (1. + 0.607 * omega_m**0.674)
    b2 = 0.238 * omega_m**0.223
    z_drag = 1345. * omega_m**0.251 / (1. + 0.659 * omega_m**0.828) * (1. + b1 * omega_b**b2)
    r_drag = 31.5 * omega_b * theta_cmb**(-4) * (1000. / (1. + z_drag))
    r_eq = 31.5 * omega_b * theta_cmb**(-4) * (1000. / (1. + z_eq))
    rs_drag = 2. / (3. * k_eq) * xnp.sqrt(6. / r_eq) * xnp.log(
        (xnp.sqrt(1. + r_drag) + xnp.sqrt(r_drag + r_eq)) / (1. + xnp.sqrt(r_eq)))
    return {'z_drag': z_drag, 'rs_drag': rs_drag * cosmo['h']}


def resample_dilated(k, column, query):
    r"""``column``, sampled on ``k``, read at ``query``, with a bounded tail off the grid.

    The dilation reads a column sampled on a fixed k grid at :math:`k / s` (or, going the
    other way, at :math:`k s`), so for :math:`s \ne 1` one end of the query range always falls
    outside the grid: above ``kmax`` when :math:`s < 1`, below ``kmin`` when :math:`s > 1`.

    ``interpax``'s ``extrap=True`` continues the last *cubic* segment there, which over the
    ~25% of the log range the ACE ``h`` box reaches is not a small error: measured over
    h in [0.50041, 0.89957] at z = 0.8 on folps' output grid, the median column's interpolation
    error in ``h`` is 1.1e-05 across the 434 of 480 grid points that are never extrapolated and
    4.2e-02 when the other 46 are included -- and every worst point, at every node count, sits
    at the last grid point. It reaches no data bin (the table is read back at
    :math:`s\,k_\mathrm{ap}`, well inside), but it is fitted, and it makes every accuracy
    metric on this emulator misleading.

    A first-order continuation in :math:`\ln k` instead, with the slope read across a window
    rather than between neighbours. Bounded, continuous with the last segment, and a smooth
    function of ``h`` -- which a clamp or a fill value would not be: those put a kink at the
    ``s`` where each component crosses the boundary, and a kink is worse for a Chebyshev
    expansion than a wrong-but-smooth value in a region nothing reads.

    The tail is not always in a region nothing reads. ``desilike``'s ``FOLPSD3PolesEmulator`` resamples the
    other way at prediction time, and ``folps.sigmas`` integrates the reconstructed rows over the
    whole grid -- so the tail lands inside the BAO damping.

    **A power law was tried here and removed.** It is the shape these columns actually have at
    the ends of the grid, and on the bispectrum multipoles over the ACE ``h`` box at 5 nodes it
    was worth median / max ``|dB/B|`` of 5.6e-06 / 4.4e-05 against the 8.2e-06 / 2.4e-04 this
    gives -- so a factor 1.5 on the median and 5 on the max, the worst point being the top edge
    of the box.

    It is also unsafe, because an exponential of a locally-estimated slope has no bound: at
    ``nk = 480`` the spacing is ``d ln k = 0.013``, so two samples differing by a factor 3 --
    which the no-wiggle ``Ploop_dt`` does near a zero crossing -- give a slope of 84, and
    exponentiating that over the 0.30 in ``ln k`` the ACE ``h`` box reaches turned a table value
    of 445 into 7.5e+13. That was fitted, leaving coefficients of 1e+13 in ``table_now.4``
    against a median of 1.8e-04, and the emulated chi2 then diverged from the exact one by 1e+15
    at every posterior sample. Clipping the slope to :math:`|{\rm d}\ln c / {\rm d}\ln k| \le 4`
    fixed that and kept the accuracy; a first-order tail cannot go wrong that way at all, which
    is why it is what ships. To take the factor 5 back, restore the power law with that clip --
    the window below is what makes the slope estimate stable, the clip is what makes the
    exponential safe, and neither is sufficient alone.
    """
    import interpax

    # The grid is detached as an abscissa, for the reason spelled out in
    # `FOLPSPTSpectrum2Poles.combine_bias_terms_spectrum3_poles`: an emulated calculator emits it
    # as a traced output whose derivative is exactly zero, and differentiating an interpolation
    # with respect to its own abscissa is NaN at a node, which `0 * NaN = NaN` then propagates
    # into every cosmological gradient. It is genuinely constant, so this is exact. `query` keeps
    # its gradient -- that is where the dilation scale enters.
    k = jax.lax.stop_gradient(jnp.asarray(k))
    column = jnp.asarray(column)
    query = jnp.asarray(query)
    lnk, lnq = jnp.log(k), jnp.log(jnp.abs(query))
    out = interpax.interp1d(jnp.clip(query, k[0], k[-1]), k, column, method='cubic')

    def tail(edge, inner, lnedge, lninner):
        return edge + (edge - inner) / (lnedge - lninner) * (lnq - lnedge)

    # The slope is read across a window, not between neighbours: a secant over one interval of a
    # fine grid measures the local noise rather than the trend, and the tail then carries it over
    # the whole extrapolated range. At nk = 480 this is 9 intervals, d ln k = 0.12 against 0.013.
    step = max(1, k.size // 50)
    return jnp.where(query < k[0], tail(column[0], column[step], lnk[0], lnk[step]),
                     jnp.where(query > k[-1],
                               tail(column[-1], column[-1 - step], lnk[-1], lnk[-1 - step]),
                               out))


def dilate(k, value, scale, axis=-1):
    r"""*value*, sampled on *k* along *axis*, read at ``k * scale`` row by row.

    The dilation itself: at fixed physical densities the transfer function in
    :math:`\mathrm{Mpc}^{-1}` does not move with :math:`h`, so a spectrum in
    :math:`(\mathrm{Mpc}/h)^3` on a grid in :math:`h/\mathrm{Mpc}` obeys
    :math:`P_h(k) = s^3 P_\mathrm{fid}(k s)`, :math:`s = h / h_\mathrm{fid}`. Dividing that out
    leaves the interpolant a reference-frame spectrum plus the physics residual -- a smooth
    function of :math:`h` -- instead of the BAO wiggles sliding through the k grid, which no
    low-order polynomial follows. The :math:`s^3` is the caller's to apply.
    """
    k, value = np.asarray(k), jnp.asarray(value)
    value = jnp.moveaxis(value, axis, -1)
    shape = value.shape
    rows = jax.vmap(lambda row: resample_dilated(k, row, k * scale))(value.reshape(-1, k.size))
    return jnp.moveaxis(rows.reshape(shape), -1, axis)
