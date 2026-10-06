r"""
Fast, vectorised gradient-stability check for the Horndeski models of mochi_class.

Purpose
-------
A linear-:math:`P(k)` emulator has no way of knowing whether the model it is being
asked about is physically viable.  mochi_class answers that question by running the
Boltzmann code and letting it abort (``stability_tests_smg`` in
``gravity_smg/background_smg.c``); at ~0.1 s per background that is far too slow to
gate an emulator call.  This module reproduces the same decision analytically, for
millions of models at a time.

What is (and is not) checked
----------------------------
mochi_class runs four tests on the background table::

    min D    >= -|D_safe_smg|      ghost, scalar
    min c_s^2 > -|cs2_safe_smg|    gradient, scalar
    min M_*^2 >= -|M2_safe_smg|    ghost, tensor
    min c_t^2 >= -|ct2_safe_smg|   gradient, tensor

with the ``*_safe_smg`` thresholds all 0 by default.  Here

* ``D = alpha_K + 3/2 alpha_B^2`` is positive whenever ``alpha_K > 0``, so the scalar
  no-ghost test is vacuous by construction and is skipped (:func:`kinetic_D` is
  provided if you want to assert it);
* ``c_t^2 = 1 + alpha_T`` and ``M_*^2 > 0`` are trivial for the two models here
  (``alpha_T = 0`` for hill_valley; ``M_*^2 = M2_ini exp(int alpha_M dlna) > 0``
  always).  :func:`stable` checks them anyway -- they cost nothing.

That leaves the **gradient instability**, which is what this module is really about.

There is a fifth condition, and it is easy to miss.  ``background_solve_smg``
(``background_smg.c:1108``) refuses outright any model whose braiding **crosses 2**,
because ``2 - alpha_B`` sits in the denominator of the perturbation equations.  It is
*not* gated by ``skip_stability_tests_smg``, so a run made with that flag set -- the
obvious way to generate training labels -- still raises on it.  For ``propto_omega`` it
is not a corner case: ``alpha_B = c_b Omega_smg`` sweeps from 0 up to
``c_b Omega_smg,0``, so every ``c_b`` above roughly 2.9 is rejected on these grounds
alone.  :func:`stable` applies it.

Mathematical stability (``skip_math_stability_smg``) is a perturbation-level test and is
not reproduced here -- and does not need to be: it is ``_TRUE_`` (i.e. skipped) by
default in ``input_default_params_smg``.

The criterion
-------------
Read off ``gravity_functions_As_from_alphas_smg`` (``gravity_smg/gravity_functions_smg.c``),
with ``bra = alpha_B``, ``run = alpha_M``, ``ten = alpha_T``, ``kin = alpha_K``:

.. code::

    D      = kin + 3/2 bra^2
    lam2   = -3/2 X_m (bra - 2 dM2/M2) - 3/2 X_de (bra - 2) + dbra/dlna
    cs2num = (bra - 2)(-bra - 2 run + 2 ten - bra ten)/2 + lam2
    c_s^2  = cs2num / D

where ``dM2 = M_*^2 - 1`` and, with ``H^2 = rho_tot`` in CLASS' units (the Friedmann
equation carries *no* factor of ``M_*^2``: for these parametrisations the background is
exactly w0waCDM, with ``rho_smg`` absorbing everything),

.. code::

    X_m  = (rho_tot,wo_smg + p_tot,wo_smg) / H^2
    X_de = (rho_smg + p_smg) / H^2 = (1 + w(a)) Omega_smg(a)

``cs2num`` as assembled here reproduces mochi_class' own ``cs2num`` column to 1e-15
relative -- it is the same expression, not an approximation.  Since ``D > 0``,
``min_a c_s^2 < 0`` if and only if ``min_a cs2num < 0``, so the whole gradient test
reduces to the sign of ``cs2num`` and **never involves alpha_K at all**.

The only approximation is therefore the *background*, and even that is exact up to the
Fermi-Dirac quadrature for massive neutrinos (:data:`_NCDM_RHO`), which is calibrated
against CLASS itself.  See below for where the measured agreement is produced.

Validation, and where it lives
------------------------------
The measured agreement with mochi_class -- pointwise ``cs2num``, the confusion matrix of
the verdict, and the displacement of the decision boundary, which is the number that
really sets the error rate -- is produced by ``validate_stability.py`` and
``validate_boundary.py``.  Those two scripts are **not** part of ``cosmoprimo``: they need
a working ``pyclass.mochiclass`` to generate ground truth, and they live with the rest of
the notes in the DESI-DR2-MG ``Stability/`` directory.  Nothing here imports them, and the
module itself is numpy-only.

Conventions
-----------
All alpha functions are in the **hi_class / mochi_class** convention, not EFTCAMB's
(EFTCAMB's braiding is ``-1/2`` of the one used here; ``cosmoprimo``'s ``heftcamb``
engine applies that internally).  Note also that EFTCAMB's own stability module uses a
different, non-equivalent set of conditions -- this module reproduces *mochi_class*.

Usage
-----
Every parameter broadcasts, so a whole prior sample goes through in one call::

    import numpy as np
    from cosmoprimo.emulators import mochiclass as mcs

    n = 1_000_000
    ok = mcs.stable_hill_valley(
        c_M=np.random.uniform(-0.5, 0.5, n),
        tau=np.random.uniform(0.5, 5., n),
        a_t=np.random.uniform(0.1, 1., n),
        r=2.,
        omega_cdm=np.random.uniform(0.10, 0.14, n),
        omega_b=0.02237, h=0.6736, w0=-0.9, wa=0.36)

``ok`` is a boolean array: ``True`` where mochi_class would run the model.
"""

import numpy as np

__all__ = ['stable', 'stable_hill_valley', 'stable_propto_omega', 'class_a_grid',
           'class_a_grid',
           'scan_hill_valley', 'scan_propto_omega',
           'min_cs2num_hill_valley', 'min_cs2num_propto_omega',
           'cs2num', 'kinetic_D', 'background', 'default_a_grid',
           'alphas_hill_valley', 'alphas_propto_omega']


# --------------------------------------------------------------------------------------
# CLASS' own physical constants (include/background.h), so that the background matches
# mochi_class rather than some other rounding of the same physics.
# --------------------------------------------------------------------------------------
_c_ = 2.99792458e8
_G_ = 6.67428e-11
_k_B_ = 1.3806504e-23
_h_P_ = 6.62606896e-34
_eV_ = 1.602176487e-19
_Mpc_over_m_ = 3.085677581282e22
_sigma_B_ = 2. * np.pi**5 * _k_B_**4 / 15. / _h_P_**3 / _c_**2

#: CLASS' ``T_ncdm_default``; chosen so that m / omega_ncdm = 93.14 eV.
T_NCDM_DEFAULT = 0.71611

#: CLASS' ``a_ini_over_a_today_default``: where the background table, and hence the
#: stability scan, starts.
A_INI = 1e-14

#: Number of rows in mochi_class' background table (uniform in ln a over [A_INI, 1]),
#: i.e. the sampling on which ``min_cs2_smg`` is actually evaluated.
N_CLASS_TABLE = 3000

#: Default number of grid points.  See :func:`default_a_grid` for why 512 non-uniform
#: points are worth more than 3000 uniform ones.
DEFAULT_NA = 512


def omega_g(T_cmb=2.7255):
    """:math:`\\Omega_\\gamma h^2`, verbatim from CLASS' ``input.c`` (line 2474)."""
    return ((4. * _sigma_B_ / _c_ * np.asarray(T_cmb, dtype='f8')**4)
            / (3. * _c_**2 * 1.e10 / _Mpc_over_m_**2 / 8. / np.pi / _G_))


# --------------------------------------------------------------------------------------
# Massive neutrinos.
#
# rho_ncdm(a) and p_ncdm(a) are pure Fermi-Dirac integrals of the single dimensionless
# combination y = a M, M = m / T_ncdm,0.  Tabulating them once, as a *universal* pair of
# functions, makes massive neutrinos exact rather than approximate at negligible cost:
#
#     Irho(y) = int_0^inf q^2 sqrt(q^2 + y^2) / (e^q + 1) dq
#     Ip(y)   = 1/3 int_0^inf q^4 / (sqrt(q^2 + y^2) (e^q + 1)) dq
#
#     rho_ncdm / rho_g = (15 / pi^4) deg (T_ncdm / T_cmb)^4 a^-4 Irho(a M)
#
# which reduces to the familiar (7/8) (T_nu/T_gamma)^4 per family as y -> 0, since
# Irho(0) = 7 pi^4 / 120.
# --------------------------------------------------------------------------------------
_IRHO_0 = 7. * np.pi**4 / 120.          # massless limit of Irho
_IP_0 = _IRHO_0 / 3.                    # massless limit of Ip
_ZETA3_FACTOR = 1.5 * 1.2020569031595943   # Irho(y) -> (3/2) zeta(3) y as y -> inf

_LOG_Y_MIN, _LOG_Y_MAX, _N_Y = -6., 8., 2048


def _build_ncdm_tables():
    r"""
    Tabulate ``Irho``, ``Ip`` and their ``y`` derivatives, by Gauss-Legendre quadrature.

    The derivatives are only needed by :func:`background` with ``derivs=True`` (i.e. by
    ``cosmoprimo.emulators.heftcamb``, which needs ``d p_tot / d ln a`` to build
    :math:`d^2\Omega_{\rm smg}/d\ln a^2`); they cost nothing to tabulate alongside.

    .. code::

        dIrho/dy = y   int q^2 / sqrt(q^2 + y^2) f dq
        dIp/dy   = -y/3 int q^4 / (q^2 + y^2)^{3/2} f dq
    """
    # The integrands are smooth and die like e^-q; 256 nodes on [0, 40] is machine
    # precision, and this runs once at import (a few ms).
    x, w = np.polynomial.legendre.leggauss(256)
    q = 20. * (x + 1.)
    wq = 20. * w
    f = 1. / (np.exp(q) + 1.)
    y = np.logspace(_LOG_Y_MIN, _LOG_Y_MAX, _N_Y)
    s = np.sqrt(q[None, :]**2 + y[:, None]**2)
    irho = ((q**2 * f * wq)[None, :] * s).sum(axis=1)
    ip = ((q**4 * f * wq)[None, :] / s).sum(axis=1) / 3.
    dirho = y * ((q**2 * f * wq)[None, :] / s).sum(axis=1)
    dip = -y / 3. * ((q**4 * f * wq)[None, :] / s**3).sum(axis=1)
    return y, irho, ip, dirho, dip


_NCDM_Y, _NCDM_RHO, _NCDM_P, _NCDM_DRHO, _NCDM_DP = _build_ncdm_tables()
_NCDM_LOGY = np.log(_NCDM_Y)
_NCDM_LOGRHO = np.log(_NCDM_RHO)
_NCDM_LOGP = np.log(_NCDM_P)


def _ncdm_integrals(y, derivs=False):
    """
    ``Irho(y), Ip(y)``, with the exact asymptotics used outside the tabulated range.

    With ``derivs=True`` also returns ``y dIrho/dy, y dIp/dy`` -- the log-derivative
    combination, which is what enters ``d rho_ncdm / d ln a``.
    """
    y = np.asarray(y, dtype='f8')
    ly = np.log(np.clip(y, 1e-300, None))
    irho = np.exp(np.interp(ly, _NCDM_LOGY, _NCDM_LOGRHO))
    ip = np.exp(np.interp(ly, _NCDM_LOGY, _NCDM_LOGP))
    # y below the table: fully relativistic. y above it: fully non-relativistic.
    lo = y < _NCDM_Y[0]
    hi = y > _NCDM_Y[-1]
    if lo.any():
        irho = np.where(lo, _IRHO_0, irho)
        ip = np.where(lo, _IP_0, ip)
    if hi.any():
        irho = np.where(hi, _ZETA3_FACTOR * y, irho)
        ip = np.where(hi, 0., ip)
    if not derivs:
        return irho, ip
    ydirho = y * np.interp(ly, _NCDM_LOGY, _NCDM_DRHO)
    ydip = y * np.interp(ly, _NCDM_LOGY, _NCDM_DP)
    if lo.any():                      # relativistic: both integrals are y-independent
        ydirho = np.where(lo, 0., ydirho)
        ydip = np.where(lo, 0., ydip)
    if hi.any():                      # Irho -> (3/2) zeta(3) y, Ip -> 0
        ydirho = np.where(hi, _ZETA3_FACTOR * y, ydirho)
        ydip = np.where(hi, 0., ydip)
    return irho, ip, ydirho, ydip


# --------------------------------------------------------------------------------------
# Background
# --------------------------------------------------------------------------------------
def default_a_grid(na=DEFAULT_NA, a_min=A_INI, a_split=1e-3, frac_early=0.25):
    r"""
    The scale-factor grid the stability scan runs on: log-spaced, but in two pieces.

    ``[a_min, 1]`` is 32 e-folds, and mochi_class covers it with 3000 points uniform in
    :math:`\ln a`.  Almost all of that range carries no structure: the alphas of both
    parametrisations are exponentially small at early times, and ``cs2num`` tends
    smoothly to zero there.  Everything that decides the verdict happens in the last few
    decades.  So ``frac_early`` of the points go on ``[a_min, a_split]`` and the rest on
    ``[a_split, 1]``.

    That is worth a lot.  Measured against the exact 3000-point grid over a broad prior
    -- mean displacement of the decision boundary, in units of the prior width::

        grid            hill_valley   propto_omega
        512 uniform        1.4e-4        1.2e-4
        512 two-part       4.9e-5        7.1e-6

    i.e. for ``propto_omega`` 512 non-uniform points are worth more than 3000 uniform
    ones, at a sixth of the cost.  :func:`class_a_grid` reproduces mochi_class' own
    sampling exactly if you want it.

    ``na`` is the total number of points, and is the main speed/fidelity dial;
    ``validate_stability.py`` measures what thinning it costs.
    """
    na = int(na)
    ne = max(2, int(na * frac_early))
    return np.concatenate([np.logspace(np.log10(a_min), np.log10(a_split), ne, endpoint=False),
                           np.logspace(np.log10(a_split), 0., na - ne)])


def class_a_grid():
    r"""mochi_class' own background sampling: 3000 points uniform in :math:`\ln a`."""
    return np.logspace(np.log10(A_INI), 0., N_CLASS_TABLE)


def background(a, h=0.6736, omega_b=0.02237, omega_cdm=0.12, w0=-1., wa=0.,
               m_ncdm=0.06, deg_ncdm=1., T_ncdm=T_NCDM_DEFAULT, N_ur=2.0328,
               T_cmb=2.7255, omega_ncdm=None, Omega_smg=None, derivs=False):
    r"""
    The dimensionless background combinations the stability criterion needs.

    Everything is a ratio to :math:`H^2`, so no unit or :math:`H_0` convention survives
    into the answer.  ``a`` broadcasts against the (array-valued) parameters; the usual
    call has parameters of shape ``(n, 1)`` and ``a`` of shape ``(na,)`` or ``(n, na)``.

    ``m_ncdm`` is the mass in eV of a **single** ncdm species with degeneracy
    ``deg_ncdm`` (``deg_ncdm=3`` gives three degenerate massive neutrinos, as CLASS
    does).  Note that ``m_ncdm=0`` is a *massless* ncdm species at ``T_ncdm``, exactly as
    it is in CLASS -- to have no ncdm species at all, pass ``deg_ncdm=0`` (and move the
    relativistic degrees of freedom into ``N_ur``).

    ``Omega_smg`` defaults to ``None``, meaning mochi_class' closure equation
    (``Omega_smg = -1`` in the ``.ini``): it is fixed by flatness.  Pass a number to
    override.

    Returns a dict with

    ``X_m``
        :math:`(\rho_{\rm tot,wo\,smg} + p_{\rm tot,wo\,smg}) / H^2`
    ``X_de``
        :math:`(\rho_{\rm smg} + p_{\rm smg}) / H^2`
    ``Omega_de``
        :math:`\rho_{\rm smg} / H^2` -- mochi_class' ``Omega_smg``, the shape function
        of the ``propto_omega`` parametrisation
    ``P_tot``
        :math:`p_{\rm tot} / H^2`, *including* smg; only needed for
        :math:`d\Omega_{\rm smg}/d\ln a`
    ``w``
        :math:`w_0 + w_a (1 - a)`

    With ``derivs=True`` three more entries appear, all :math:`d/d\ln a`:
    ``dP_tot``, ``dOmega_de`` and ``d2Omega_de``.  ``cosmoprimo.emulators.heftcamb`` needs
    them because EFTCAMB's :math:`\Omega`, and hence its gradient term, depends on
    :math:`\alpha_T'' \propto \Omega_{\rm smg}''`.
    """
    a = np.asarray(a, dtype='f8')
    h, omega_b, omega_cdm = (np.asarray(x, dtype='f8') for x in (h, omega_b, omega_cdm))
    w0, wa = np.asarray(w0, dtype='f8'), np.asarray(wa, dtype='f8')
    m_ncdm = np.asarray(m_ncdm, dtype='f8')

    h2 = h**2
    Om_g0 = omega_g(T_cmb) / h2
    Om_ur0 = N_ur * 7. / 8. * (4. / 11.)**(4. / 3.) * Om_g0
    Om_m0 = (omega_b + omega_cdm) / h2

    # ncdm.  M = m / T_ncdm,0 exactly as CLASS' input.c: m_in_eV / k_B * eV / T_ncdm / T_cmb.
    M = m_ncdm / _k_B_ * _eV_ / T_ncdm / T_cmb
    ncdm_pref = 15. / np.pi**4 * deg_ncdm * T_ncdm**4 * Om_g0
    if omega_ncdm is None:
        Om_ncdm0 = ncdm_pref * _ncdm_integrals(M)[0]
    else:
        Om_ncdm0 = np.asarray(omega_ncdm, dtype='f8') / h2
        # Renormalise the shape to hit the requested omega_ncdm today, as CLASS does
        # via fnu_factor.
        ncdm_pref = ncdm_pref * Om_ncdm0 / (ncdm_pref * _ncdm_integrals(M)[0])

    if Omega_smg is None:
        Om_de0 = 1. - Om_g0 - Om_ur0 - Om_ncdm0 - Om_m0
    else:
        Om_de0 = np.asarray(Omega_smg, dtype='f8')

    am4 = a**-4
    rho_g = Om_g0 * am4
    rho_ur = Om_ur0 * am4
    nc = _ncdm_integrals(a * M, derivs=derivs)
    rho_ncdm = ncdm_pref * am4 * nc[0]
    p_ncdm = ncdm_pref * am4 * nc[1]
    rho_m = Om_m0 * a**-3

    w = w0 + wa * (1. - a)
    rho_de = Om_de0 * a**(-3. * (1. + w0 + wa)) * np.exp(3. * wa * (a - 1.))
    p_de = w * rho_de

    p_rad = (rho_g + rho_ur) / 3.
    rho_wo = rho_g + rho_ur + rho_ncdm + rho_m
    p_wo = p_rad + p_ncdm
    E2 = rho_wo + rho_de

    Om_de = rho_de / E2
    P_tot = (p_wo + p_de) / E2
    toret = {'X_m': (rho_wo + p_wo) / E2,
             'X_de': (rho_de + p_de) / E2,
             'Omega_de': Om_de,
             'P_tot': P_tot,
             'w': w,
             'E2': E2}
    if not derivs:
        return toret

    # d/dln a of each pressure.  Radiation goes like a^-4 so dp/dlna = -4p; the ncdm
    # shape carries the extra y dI/dy; the fluid picks up dw/dlna = -a w_a.
    dw = -a * wa
    dp_rad = -4. * p_rad
    dp_ncdm = -4. * p_ncdm + ncdm_pref * am4 * nc[3]
    dp_de = (dw + w * (-3.) * (1. + w)) * rho_de
    # dln E^2 / dlna = dln rho_tot / dlna = -3 (1 + P_tot)
    toret['dP_tot'] = (dp_rad + dp_ncdm + dp_de) / E2 + 3. * P_tot * (1. + P_tot)
    # Omega_de = rho_de / rho_tot, so dlnOmega_de/dlna = -3(1+w) + 3(1+P_tot).
    toret['dOmega_de'] = 3. * Om_de * (P_tot - w)
    toret['d2Omega_de'] = 3. * (toret['dOmega_de'] * (P_tot - w)
                                + Om_de * (toret['dP_tot'] - dw))
    return toret


# --------------------------------------------------------------------------------------
# alpha functions
# --------------------------------------------------------------------------------------
def alphas_hill_valley(a, c_M, tau, a_t, r=2., M2_ini=1.):
    r"""
    The hill/valley (No Slip Gravity) alphas of `arXiv:1904.12903
    <https://arxiv.org/abs/1904.12903>`_, in closed form:

    .. math::
        \alpha_M = c_M \tanh u\ {\rm sech}^2 u, \quad u = \tfrac{\tau}{2}\ln(a/a_t),
        \quad \alpha_B = -r\,\alpha_M, \quad \alpha_T = 0,
        \quad M_\ast^2 = M_{\ast,\rm ini}^2 e^{-(c_M/\tau)\,{\rm sech}^2 u}.

    ``d alpha_B / d ln a`` is differentiated analytically, so nothing here is sampled or
    splined.  The ``sech``/``tanh`` are built from ``exp(-2|u|)`` exactly as
    ``gravity_models_hill_valley_smg`` does, which keeps them exact out to
    :math:`a = 10^{-14}` where ``cosh u`` would have overflowed.

    Returns ``(alpha_B, alpha_M, alpha_T, dalpha_B/dlna, M2)``.
    """
    u = 0.5 * tau * np.log(a / a_t)
    x2 = np.exp(-2. * np.abs(u))
    opx2 = 1. + x2
    sech2 = 4. * x2 / opx2**2
    tanh_u = np.sign(u) * (1. - x2) / opx2

    alpha_M = c_M * tanh_u * sech2
    # d/dlna (tanh u sech^2 u) = (tau/2) sech^2 u (sech^2 u - 2 tanh^2 u)
    dalpha_M = c_M * (0.5 * tau) * sech2 * (sech2 - 2. * tanh_u**2)

    alpha_B = -r * alpha_M
    M2 = M2_ini * np.exp(-(c_M / tau) * sech2)
    return alpha_B, alpha_M, np.zeros_like(alpha_B), -r * dalpha_M, M2


def alphas_propto_omega(a, lna, bg, c_b, c_m, c_t=0., M2_ini=1.):
    r"""
    The ``propto_omega`` alphas, :math:`\alpha_i = c_i\,\Omega_{\rm smg}(a)` with
    :math:`\Omega_{\rm smg} = \rho_{\rm smg}/\rho_{\rm tot}` (*not* normalised to
    :math:`\Omega_{\rm smg,0}`, matching ``gravity_models_smg.c``).

    Two derivatives are needed and both are analytic in the background:

    .. math::
        \frac{d\Omega_{\rm smg}}{d\ln a} = 3\,\Omega_{\rm smg}
            \left(\frac{p_{\rm tot}}{H^2} - w\right),

    from :math:`d\ln\rho_{\rm smg}/d\ln a = -3(1+w)` and
    :math:`d\ln\rho_{\rm tot}/d\ln a = -3(1 + p_{\rm tot}/H^2)`.

    :math:`M_\ast^2` has no closed form here -- mochi_class integrates
    :math:`d\ln M_\ast^2/d\ln a = \alpha_M` from ``a_ini`` with
    :math:`M_\ast^2(a_{\rm ini}) = M^2_{\ast,\rm ini}` -- but the integral
    :math:`\int \Omega_{\rm smg}\,d\ln a` is independent of ``c_m``, so one cumulative
    trapezoid over the (sorted) grid does it.  ``lna`` must be ``log(a)``, increasing
    along the last axis.

    Returns ``(alpha_B, alpha_M, alpha_T, dalpha_B/dlna, M2)``.
    """
    Om = bg['Omega_de']
    alpha_B = c_b * Om
    alpha_M = c_m * Om
    alpha_T = c_t * Om
    dOm = 3. * Om * (bg['P_tot'] - bg['w'])

    # cumulative trapezoid of Omega_smg d ln a, starting at 0 on the first grid point
    dl = np.diff(lna, axis=-1)
    integ = np.concatenate([np.zeros(Om.shape[:-1] + (1,)),
                            np.cumsum(0.5 * (Om[..., 1:] + Om[..., :-1]) * dl, axis=-1)],
                           axis=-1)
    M2 = M2_ini * np.exp(c_m * integ)
    return alpha_B, alpha_M, alpha_T, c_b * dOm, M2


# --------------------------------------------------------------------------------------
# The criterion itself
# --------------------------------------------------------------------------------------
def cs2num(alpha_B, alpha_M, alpha_T, dalpha_B_dlna, M2, X_m, X_de):
    """
    mochi_class' ``cs2num``, i.e. the numerator of the scalar sound speed.

    Identical (to 1e-15 relative) to the ``cs2num`` column of the background table; see
    ``gravity_functions_As_from_alphas_smg``.  Since ``D = alpha_K + 3/2 alpha_B^2 > 0``
    for ``alpha_K > 0``, the sign of this *is* the sign of :math:`c_s^2`.
    """
    dM2 = M2 - 1.
    lam2 = (-1.5 * X_m * (alpha_B - 2. * dM2 / M2)
            - 1.5 * X_de * (alpha_B - 2.) + dalpha_B_dlna)
    return (alpha_B - 2.) * (-alpha_B - 2. * alpha_M
                             + 2. * alpha_T - alpha_B * alpha_T) / 2. + lam2


def kinetic_D(alpha_K, alpha_B):
    """``D = alpha_K + 3/2 alpha_B^2``; the scalar no-ghost test is ``D >= 0``."""
    return alpha_K + 1.5 * alpha_B**2


# --------------------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------------------
_COSMO_KEYS = ('h', 'omega_b', 'omega_cdm', 'w0', 'wa', 'm_ncdm', 'deg_ncdm',
               'T_ncdm', 'N_ur', 'T_cmb', 'omega_ncdm', 'Omega_smg')

#: Per-chunk size of the (n_models, n_a) work array.  4e6 doubles is ~32 MB, which keeps
#: the whole scan in L3/DRAM bandwidth rather than swapping.
_CHUNK_ELEMENTS = 4_000_000


def _broadcast(names, values):
    """Broadcast a set of parameters to a common shape and flatten to ``(n,)`` each."""
    arrays = [np.asarray(v, dtype='f8') for v in values]
    shape = np.broadcast_shapes(*[x.shape for x in arrays]) if arrays else ()
    flat = [np.broadcast_to(x, shape).reshape(-1) for x in arrays]
    return dict(zip(names, flat)), shape, int(np.prod(shape, dtype='i8'))


def _refined_grid(a_grid, a_t, tau, u_max=8., n_refine=96):
    """
    Add a per-model window of points around the hill/valley transition.

    ``alpha_M`` lives entirely in ``|u| <~ 8`` (``sech^2 u = 4 e^{-2|u|}``), a window
    whose width in ``ln a`` is ``4 u_max / tau`` -- so for large ``tau`` it is narrower
    than the global grid spacing and would otherwise be stepped straight over.  The
    window is clipped to ``[A_INI, 1]``, the range mochi_class actually scans; duplicate
    points at the clip are harmless because only the minimum over the axis is used.
    """
    u = np.linspace(-u_max, u_max, int(n_refine))
    a_ref = np.clip(a_t[:, None] * np.exp(2. * u[None, :] / tau[:, None]), A_INI, 1.)
    return np.concatenate([np.broadcast_to(a_grid, (a_t.size, a_grid.size)), a_ref], axis=-1)


def _scan(model, par, cosmo_keys, a_grid, refine, n_refine, chunk):
    """
    Sweep the ``a`` grid once and return every quantity mochi_class reduces over it.

    One pass, because the five rejection conditions all read the same alphas: it would
    be wasteful (and easy to let drift) to recompute them per test.
    """
    n = len(next(iter(par.values())))
    lna = np.log(a_grid)
    width = a_grid.size + (n_refine if refine else 0)
    nchunk = chunk or max(1, _CHUNK_ELEMENTS // width)

    out = {k: np.empty(n) for k in ('min_cs2num', 'min_bra', 'max_bra', 'min_ten', 'min_M2')}
    for i in range(0, n, nchunk):
        sl = slice(i, min(i + nchunk, n))
        p = {k: v[sl][:, None] for k, v in par.items()}
        bgkw = {k: p[k] for k in cosmo_keys}
        if model == 'hill_valley':
            a = (_refined_grid(a_grid, par['a_t'][sl], par['tau'][sl], n_refine=n_refine)
                 if refine else a_grid)
            bg = background(a, **bgkw)
            al = alphas_hill_valley(a, p['c_M'], p['tau'], p['a_t'], p['r'], p['M2_ini'])
        else:
            bg = background(a_grid, **bgkw)
            al = alphas_propto_omega(a_grid, lna, bg, p['c_b'], p['c_m'], p['c_t'],
                                     p['M2_ini'])
        aB, aM, aT, daB, M2 = al
        out['min_cs2num'][sl] = cs2num(aB, aM, aT, daB, M2, bg['X_m'], bg['X_de']).min(axis=-1)
        out['min_bra'][sl] = np.broadcast_to(aB, bg['X_m'].shape).min(axis=-1)
        out['max_bra'][sl] = np.broadcast_to(aB, bg['X_m'].shape).max(axis=-1)
        out['min_ten'][sl] = np.broadcast_to(aT, bg['X_m'].shape).min(axis=-1)
        out['min_M2'][sl] = np.broadcast_to(M2, bg['X_m'].shape).min(axis=-1)
    return out


def _prepare(model, args, cosmo, na, a_grid):
    bad = set(cosmo) - set(_COSMO_KEYS)
    if bad:
        raise TypeError('unexpected argument(s) {}; expected model parameters or one of {}'
                        .format(sorted(bad), list(_COSMO_KEYS)))
    names = list(args) + list(cosmo)
    par, shape, n = _broadcast(names, list(args.values()) + list(cosmo.values()))
    grid = default_a_grid(na) if a_grid is None else np.asarray(a_grid, dtype='f8')
    return par, shape, n, grid


def scan_hill_valley(c_M, tau, a_t, r=2., M2_ini=1., na=DEFAULT_NA, a_grid=None,
                     refine=True, n_refine=96, chunk=None, **cosmo):
    r"""
    Every quantity mochi_class minimises over ``a``, for the hill/valley parametrisation.

    Returns a dict of arrays -- ``min_cs2num``, ``min_bra``, ``max_bra``, ``min_ten``,
    ``min_M2`` -- from a single sweep of the grid.  :func:`stable_hill_valley` is a thin
    wrapper; use this directly when you want the margins rather than the verdict (for
    instance to train a classifier on a smooth target rather than a boolean).
    """
    args = dict(c_M=c_M, tau=tau, a_t=a_t, r=r, M2_ini=M2_ini)
    par, shape, n, grid = _prepare('hill_valley', args, cosmo, na, a_grid)
    out = _scan('hill_valley', par, list(cosmo), grid, refine, n_refine, chunk)
    return {k: (v.reshape(shape) if shape else v[0]) for k, v in out.items()}


def scan_propto_omega(c_b, c_m, c_t=0., M2_ini=1., na=DEFAULT_NA, a_grid=None,
                      chunk=None, **cosmo):
    r"""
    Every quantity mochi_class minimises over ``a``, for ``propto_omega``.

    No per-model grid refinement is offered: :math:`\Omega_{\rm smg}(a)` is smooth and
    slowly varying, so the global log grid resolves it everywhere.  The grid must stay
    sorted because :math:`M_\ast^2` comes from a cumulative integral along it.
    """
    args = dict(c_b=c_b, c_m=c_m, c_t=c_t, M2_ini=M2_ini)
    par, shape, n, grid = _prepare('propto_omega', args, cosmo, na, a_grid)
    out = _scan('propto_omega', par, list(cosmo), grid, False, 0, chunk)
    return {k: (v.reshape(shape) if shape else v[0]) for k, v in out.items()}


def min_cs2num_hill_valley(c_M, tau, a_t, r=2., M2_ini=1., **kwargs):
    r"""
    :math:`\min_a` ``cs2num`` for the hill/valley (No Slip Gravity) parametrisation.

    ``mochi_class`` takes these as
    ``parameters_smg = [alpha_K, c_M, tau, a_t, r, M2_ini]``; ``alpha_K`` does not enter
    the gradient test (see the module docstring) and is not an argument here.

    All model and cosmological parameters broadcast against one another; the result has
    their common shape.  Negative means mochi_class would abort with *"Gradient
    instability for scalar field perturbations"*.  Note this is only one of the five
    conditions mochi_class applies -- :func:`stable_hill_valley` applies all of them.

    Extra keyword arguments are passed to :func:`background` (``h``, ``omega_b``,
    ``omega_cdm``, ``w0``, ``wa``, ``m_ncdm``, ...) or control the grid (``na``,
    ``a_grid``, ``refine``, ``n_refine``, ``chunk``).
    """
    return scan_hill_valley(c_M, tau, a_t, r=r, M2_ini=M2_ini, **kwargs)['min_cs2num']


def min_cs2num_propto_omega(c_b, c_m, c_t=0., M2_ini=1., **kwargs):
    r"""
    :math:`\min_a` ``cs2num`` for ``propto_omega`` (:math:`\alpha_i = c_i \Omega_{\rm smg}(a)`).

    ``mochi_class`` takes these as
    ``parameters_smg = [alpha_K, c_b, c_m, c_t, M2_ini]``; ``c_k`` (:math:`\alpha_K`)
    does not enter the gradient test and is not an argument here.  As above, this is one
    condition of five; :func:`stable_propto_omega` applies all of them.
    """
    return scan_propto_omega(c_b, c_m, c_t=c_t, M2_ini=M2_ini, **kwargs)['min_cs2num']


def _verdict(s, M2_ini, alpha_K):
    """
    Apply mochi_class' rejection conditions to the output of a scan.

    Five conditions, four of them from ``stability_tests_smg`` and one from
    ``background_solve_smg``:

    ``min c_s^2 >= 0``
        gradient stability of the scalar.  ``sign(c_s^2) = sign(cs2num)`` because
        ``D > 0``; see :func:`cs2num`.
    ``min D >= 0``
        no scalar ghost.  Guaranteed by ``alpha_K >= 0``, checked only if ``alpha_K``
        was supplied.
    ``min M_*^2 >= 0``
        no tensor ghost.
    ``min c_t^2 = 1 + min alpha_T >= 0``
        gradient stability of the tensors.
    ``alpha_B does not cross 2``
        ``class_test_except`` at ``background_smg.c:1108``.  This one is **not** gated by
        ``skip_stability_tests_smg``: mochi_class refuses the model whatever that flag
        says, because ``2 - alpha_B`` sits in the denominator of the perturbation
        equations.  It is easy to miss when generating labels, since a run with
        ``skip_stability_tests_smg='yes'`` still raises on it.
    """
    ok = s['min_cs2num'] >= 0.
    ok = np.logical_and(ok, s['min_M2'] >= 0.)
    ok = np.logical_and(ok, 1. + s['min_ten'] >= 0.)
    ok = np.logical_and(ok, ~((s['min_bra'] < 2.) & (s['max_bra'] > 2.)))
    ok = np.logical_and(ok, np.asarray(M2_ini, dtype='f8') > 0.)
    if alpha_K is not None:
        ok = np.logical_and(ok, np.asarray(alpha_K, dtype='f8') >= 0.)
    return ok


def stable_hill_valley(c_M, tau, a_t, r=2., M2_ini=1., alpha_K=None, **kwargs):
    """
    ``True`` where mochi_class would run the model, ``False`` where it would abort.

    Applies all five of mochi_class' rejection conditions; see :func:`_verdict` for the
    list and for which ones are non-trivial here.

    ``alpha_K`` is optional and only ever used to reject ``alpha_K < 0``.  That is not a
    stylistic choice: the gradient verdict is the sign of ``cs2num``, which equals the
    sign of ``c_s^2 = cs2num / D`` **only while** ``D = alpha_K + 3/2 alpha_B^2 > 0``.
    With ``alpha_K >= 0`` that holds identically and both the ghost and gradient tests
    are exact.  With ``alpha_K < 0``, ``D`` can change sign and this function is no
    longer a faithful reproduction of mochi_class, so such models are reported unstable.
    """
    return _verdict(scan_hill_valley(c_M, tau, a_t, r=r, M2_ini=M2_ini, **kwargs),
                    M2_ini, alpha_K)


def stable_propto_omega(c_b, c_m, c_t=0., M2_ini=1., alpha_K=None, **kwargs):
    """
    ``True`` where mochi_class would run the model, ``False`` where it would abort.

    As :func:`stable_hill_valley`, including the ``alpha_K`` caveat.  Two conditions that
    are trivial for hill/valley bite here: ``alpha_T = c_t Omega_smg`` is not identically
    zero, so ``c_t^2 = 1 + alpha_T >= 0`` is real; and ``alpha_B = c_b Omega_smg`` runs
    from 0 up to ``c_b Omega_smg,0``, so any ``c_b > 2 / Omega_smg,0`` (about 2.9) makes
    the braiding cross 2 and is refused outright.
    """
    return _verdict(scan_propto_omega(c_b, c_m, c_t=c_t, M2_ini=M2_ini, **kwargs),
                    M2_ini, alpha_K)


def stable(gravity_model, parameters_smg, **cosmo):
    """
    Dispatch on mochi_class' own ``gravity_model`` / ``parameters_smg`` pair.

    ``parameters_smg`` is an array whose **last** axis holds the parameters in
    mochi_class' order, so a prior sample of shape ``(n, 6)`` goes straight in::

        # parameters_smg = [alpha_K, c_M, tau, a_t, r, M2_ini]
        ok = mgs.stable('hill_valley', theta, h=h, omega_cdm=omega_cdm, w0=w0, wa=wa)

        # parameters_smg = [alpha_K, c_b, c_m, c_t, M2_ini]
        ok = mgs.stable('propto_omega', theta, h=h, omega_cdm=omega_cdm, w0=w0, wa=wa)

    ``'no_slip_gravity'`` is accepted as an alias of ``'hill_valley'``, as in
    ``gravity_models_smg.c``.  Extra keyword arguments go to :func:`background`, plus
    ``na`` / ``a_grid`` / ``refine`` / ``chunk``.
    """
    p = np.asarray(parameters_smg, dtype='f8')
    if gravity_model in ('hill_valley', 'no_slip_gravity'):
        if p.shape[-1] != 6:
            raise ValueError('hill_valley expects parameters_smg = [alpha_K, c_M, tau, '
                             'a_t, r, M2_ini] on the last axis, got {}'.format(p.shape))
        aK, c_M, tau, a_t, r, M2_ini = (p[..., i] for i in range(6))
        return stable_hill_valley(c_M, tau, a_t, r=r, M2_ini=M2_ini, alpha_K=aK, **cosmo)
    if gravity_model == 'propto_omega':
        if p.shape[-1] != 5:
            raise ValueError('propto_omega expects parameters_smg = [alpha_K, c_b, c_m, '
                             'c_t, M2_ini] on the last axis, got {}'.format(p.shape))
        aK, c_b, c_m, c_t, M2_ini = (p[..., i] for i in range(5))
        return stable_propto_omega(c_b, c_m, c_t=c_t, M2_ini=M2_ini, alpha_K=aK, **cosmo)
    raise ValueError("gravity_model must be 'hill_valley' ('no_slip_gravity') or "
                     "'propto_omega', got {!r}".format(gravity_model))
