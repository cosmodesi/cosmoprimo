r"""
Fast, vectorised stability check for the Horndeski models of HEFTCAMB / EFTCAMB.

The companion of :mod:`mochiclass_stability`, for the other engine.  Same purpose --
gate a linear-:math:`P(k)` emulator without running the Boltzmann code -- and the same
two parametrisations, but a **different set of conditions**, because EFTCAMB's stability
module is not a re-spelling of mochi_class'.  If the emulator is trained on ``heftcamb``,
this is the gate to use; if on ``mochiclass``, use the other one.  They do not always
agree, and the differences are structural, not numerical:

===========================  ====================  ========================
condition                    mochi_class           HEFTCAMB
===========================  ====================  ========================
scalar gradient              ``c_s^2 >= 0``        ``EFT_gradient >= -tol``
scalar ghost                 ``D >= 0``            ``EFT_kinetic >= -tol``
tensor ghost                 ``M_*^2 >= 0``        ``EFTAT >= -eps``
tensor gradient              ``1 + alpha_T >= 0``  ``EFTDT >= -eps``
positive Newton constant     --                    ``1 + Omega > 0``
``alpha_B`` may not cross 2  **yes**               --
scanned range in ``a``       ``[1e-14, 1]``        ``[1e-8, 1]``
tolerance                    exactly 0             relative, see below
===========================  ====================  ========================

The tolerance matters.  EFTCAMB compares against
``tol = EFTCAMB_stability_threshold * adotoa^2 (1 + Omega)^2`` (default threshold 1e-8)
rather than against zero, because ``EFT_kinetic`` and ``EFT_gradient`` are assembled from
:math:`O(\mathcal{H}^2 (1+\Omega)^2)` terms that cancel down to :math:`O(\Omega_{\rm DE})`;
at early times the difference is round-off and its sign is meaningless.  The tensor tests
compare against the bare threshold, which is a units inconsistency in the upstream code
but is reproduced here because the point is to predict what HEFTCAMB does.

What is *not* reproduced: ``EFT_ghost_math_stability``, ``EFT_mass_math_stability``,
``EFT_mass_stability``, ``EFT_additional_priors``, ``EFT_positivity_bounds`` and
``EFT_minkowski_limit``.  All six are switched off by ``cosmoprimo``'s ``heftcamb``
engine (and the last four are off in EFTCAMB itself), so with the default settings they
never fire.  :func:`stable` will tell you if you ask for a configuration where that
assumption breaks.

The criterion
-------------
Read off ``EFTCAMBModelComputeStabilityFactors`` (``06_abstract_EFTCAMB_model.f90``) and
the RPH dictionary of ``007p2_RPH.f90``.  With :math:`\mathcal{H}` = ``adotoa`` = ``aH/c``
and primes now meaning :math:`d/da` (EFTCAMB's convention, not ``d/dln a``):

.. code::

    Omega   = M2 - 1 + alpha_T M2                     (1 + Omega = M2 (1 + alpha_T))
    c       = (H^2 - Hdot)(Omega + a Omega'/2) - (a H)^2 Omega''/2 + grhov_t (1 + w)/2
    Gamma1  = (alpha_K M2 H^2 - 2 c) / (4 h0^2 a^2)
    Gamma2  = (2 alpha_B^EFT M2 - a Omega') H / (h0 a)
    Gamma3  = -alpha_T M2,  Gamma4 = -Gamma3,  Gamma5 = Gamma3/2,  Gamma6 = 0
    EFTAT   = 1 + Omega - Gamma4 = M2
    EFTDT   = 1 + Omega          = M2 (1 + alpha_T)

with ``alpha_B^EFT = -alpha_B(hi_class)/2`` (:data:`BRAIDING_EFTCAMB_OVER_HICLASS`), and
``EFT_kinetic`` / ``EFT_gradient`` the polynomials transcribed in :func:`eft_kinetic` and
:func:`eft_gradient`.

Everything is analytic: the background is exactly w0waCDM (shared with
:mod:`mochiclass_stability`, and checked against HEFTCAMB's own ``adotoa`` to 8e-7 and
``grhov_t`` to 2e-6), and the alphas and their ``a``-derivatives are closed form.

Validating this against the code
--------------------------------
``EFTCAMB_GetEFTFunctions`` -- what ``cosmoprimo`` exposes as ``get_eft_functions`` --
used to call ``compute_background_EFT_functions`` **before** ``compute_adotoa`` on the
designer branch (its two branches were swapped relative to ``EFTStabilityComputation``), so
the ``EFTc`` it reported was evaluated at ``adotoa = Hdot = grhov_t = 0``, i.e. identically
zero.  ``EFTGamma1V`` and ``EFT_gradient`` inherited that and were wrong by factors of
10--30; ``EFT_kinetic`` was immune because ``EFTc`` cancels there against
``8 Gamma1 (1 + Omega)``.  It never touched a verdict -- the stability check has always
used the correct order -- but it made the dump useless for checking the gradient.

Fixed in ``heftcamb_upstream_fixes.patch`` (a Fortran patch against ``HEFTCAMB_fullshape``,
kept outside ``cosmoprimo`` -- see `Companion files`_ below), which also adds the missing
``compute_tensor_factors`` call (``EFTAT``, ``EFTBT``, ``EFTDT`` were reported as zero).
On a patched build the dump is usable for all of it, and this module agrees with it to

* ``EFT_kinetic`` ~2e-6, ``EFTAT`` / ``EFTDT`` 3e-8
* ``EFTc`` / ``EFT_gradient`` ~7e-3 -- background-limited, not a discrepancy: rebuilding
  ``EFTc`` from the dump's *own* ``Omega``, ``adotoa`` and ``Hdot`` reproduces it to 4e-16,
  and the residual is ``Omega = M2 - 1`` being small where ``adotoa^2`` is large.

``EFTpiA1 ... EFTpiE`` are still reported as zero: ``compute_pi_factors`` needs
``eft_cache%k``, which this routine has no notion of.  Against an *unpatched* library
``eft_kinetic`` here differs by up to 48%, by design -- see :func:`eft_kinetic`.  Verdicts
agree either way, which is checked.

A bug in the engine, now fixed
------------------------------
Part of the ``(w0, wa)`` plane used to be silently unusable on this HEFTCAMB build.
``EFTCAMBRPHSolveDesignerEquations`` (``007p2_RPH.f90``) integrated ``rho_DE`` as a
**linear** variable: ``y(1) = rho_DE / rho_DE(a_ini)``, which runs from 1 down to
``a_ini**(3(1+w0+wa)) * exp(3 wa)`` between ``a_ini = 1e-8`` and ``a = 1``.  Whenever that
fell below the solver's fixed ``atol = 1e-16``, DLSODA stopped controlling it and drove it
to zero: the dark energy vanished at late times and the run completed -- without raising --
at ``H0 = sqrt(Omega_m) H0``.  Measured before the fix, on ``w0 = -0.634, wa = +0.478``:
``H0 = 38.2`` where 67.36 was requested.  It was not the DLSODA step cap (already 100000),
and an earlier ``rhov_norm`` patch did not cure it -- that rescales the *start* of the
second pass, which moves the underflow to its end.

Fixed here by integrating the log instead: ``y(1) = log(rho_DE/rho_DE(a_ini))``, obeying
``ydot = -3(1+w(a))``, which has no ``y`` on the right-hand side, cannot underflow, spans
only O(50), and is decoupled from the rest of the system.  The physical density is
``grhov0*exp(y(1) - rhov_lognorm)*a**2``, exact at ``a = 1`` by construction.  The second
integration pass also no longer downgrades ``istate == -1`` to a warning, so this class of
failure cannot complete silently again.

After the fix, ``H0`` is correct to <= 4.3e-10 over the whole box, and the previously
broken models are all labelled correctly.  The diff is saved as
``heftcamb_upstream_fixes.patch``, with the pre-patch source and library kept beside the
patched build in ``HEFTCAMB_fullshape/.stability_backup/``.

**Applying it fixes one build.** Anyone on an unpatched HEFTCAMB still has the bug, and
``cosmoprimo``'s ``_designer_normalisation_is_reliable`` probe does not detect it (it tests
``w0 = -0.9, wa = -0.5``, which converges either way).  ``validate_heftcamb.py`` keeps its
own check -- comparing HEFTCAMB's ``H0`` against the requested one -- as a regression guard;
it is a cheap thing to keep in an emulator training loop for the same reason.

Usage
-----
::

    import numpy as np
    from cosmoprimo import heftcamb_stability as hs

    ok = hs.stable_hill_valley(
        c_M=np.random.uniform(-0.5, 0.5, n), tau=..., a_t=..., r=2., alpha_K=1e-4,
        h=0.6736, omega_b=0.02237, omega_cdm=0.12, w0=-0.9, wa=0.36)

Alphas are in the **hi_class / mochi_class** convention throughout, exactly as
``cosmoprimo``'s ``heftcamb`` engine takes them, so the same numbers go into this module
and into :mod:`mochiclass_stability`.

.. _Companion files:

Companion files
---------------
Two files referred to above are **not** shipped with ``cosmoprimo`` -- they live with the
rest of the notes in the DESI-DR2-MG ``Stability/`` directory:

* ``heftcamb_upstream_fixes.patch``, the Fortran diff against ``HEFTCAMB_fullshape``:
  swapped ``EFTCAMB_GetEFTFunctions`` branches, missing ``compute_tensor_factors``, the
  no-op third pass of ``EFTCAMB_Stability_Check`` (:func:`eftcamb_a_grid`), and the
  designer ``rho_DE`` underflow.
* ``validate_heftcamb.py``, which generates ground truth by asking the ``heftcamb`` engine
  itself and reports the confusion matrix, the pointwise ``EFT_kinetic`` agreement, and how
  often HEFTCAMB and mochi_class disagree with each other.

This module needs neither -- it is numpy-only and reads nothing from a HEFTCAMB build.
Against an *unpatched* library the first three fixes cost it nothing: the first two touch
only the diagnostic dump, and the third changed 0 / 4000 verdicts.  The ``rho_DE``
underflow is different in kind -- it makes the engine return a ``P(k)`` at the wrong
``H0`` rather than raise -- so on an unpatched build the disagreements it causes are the
engine's, not this module's.
"""

import numpy as np

from . import mochiclass_stability as mcs
from .mochiclass_stability import background, alphas_hill_valley

__all__ = ['stable', 'stable_hill_valley', 'stable_propto_omega',
           'scan_hill_valley', 'scan_propto_omega',
           'eft_functions', 'eft_kinetic', 'eft_gradient',
           'default_a_grid', 'eftcamb_a_grid']


#: ``alpha_B(EFTCAMB) = -0.5 alpha_B(hi_class)``; see ``cosmoprimo/heftcamb.py``.
BRAIDING_EFTCAMB_OVER_HICLASS = -0.5

#: ``EFTCAMB_stability_time``: where the scan starts.  1e-8 is what ``cosmoprimo`` sets
#: (EFTCAMB's own default of 1e-10 precedes the designer background grid).
A_START = 1e-8

#: ``EFTCAMB_stability_threshold``: the relative tolerance on the kinetic and gradient
#: terms.  EFTCAMB's default.
STABILITY_THRESHOLD = 1e-8

#: ``indMax`` in ``EFTCAMB_Stability_Check``: points per sampling pass.
N_EFTCAMB_SAMPLE = 1000

#: Default number of grid points; see :func:`default_a_grid`.
DEFAULT_NA = 512

#: Mpc per unit ``h`` in ``H0``: ``h0_Mpc = h / _C_OVER_100``.
_C_OVER_100 = 2997.92458458


def eftcamb_a_grid(a_start=A_START, n=N_EFTCAMB_SAMPLE):
    r"""
    EFTCAMB's own stability sampler, reproduced exactly.

    ``EFTCAMB_Stability_Check`` makes three passes of ``n`` points over ``[a_start, 1]``:
    a linear one, a log one bunched at ``a_start``
    (:math:`a = a_{\rm start} + (1 - a_{\rm start}) 10^{y}`, :math:`y \in [-10, 0]`), and a
    third bunched at ``a = 1`` (:math:`a = 1 + (a_{\rm start} - 1) 10^{y}`).

    The third pass was a no-op upstream -- ``y`` was never reassigned inside that loop, so
    it kept the value 0 left by the previous one and every iteration re-tested ``a_start``.
    Fixed in ``heftcamb_upstream_fixes.patch`` and included here.  It is inert in practice:
    the linear pass already reaches ``a = 1`` and the terms are smooth there, so restoring
    the 1000 extra points changed 0 / 4000 verdicts on a broad prior.
    """
    lin = np.linspace(a_start, 1., int(n))
    y = np.linspace(-10., 0., int(n))
    log_start = a_start + (1. - a_start) * 10.**y
    log_end = 1. + (a_start - 1.) * 10.**y
    return np.unique(np.concatenate([lin, log_start, log_end]))


def default_a_grid(na=DEFAULT_NA, a_start=A_START, a_split=1e-3, frac_early=0.25):
    """
    A cheaper stand-in for :func:`eftcamb_a_grid`, in the spirit of
    :func:`mochiclass_stability.default_a_grid`: log-spaced in two pieces, dense where the
    structure is.

    ``validate_heftcamb.py`` measures what it costs relative to the exact sampler.
    """
    na = int(na)
    ne = max(2, int(na * frac_early))
    return np.concatenate([np.logspace(np.log10(a_start), np.log10(a_split), ne, endpoint=False),
                           np.logspace(np.log10(a_split), 0., na - ne)])


# --------------------------------------------------------------------------------------
# RPH alphas and their a-derivatives
#
# EFTCAMB differentiates with respect to a, not ln a, so everything below carries the
# 1/a factors explicitly.  With u = (tau/2) ln(a/a_t) and du/dln a = tau/2,
#
#     alpha_M      = c_M tanh u sech^2 u
#     dalpha_M/dln a = c_M (tau/2) sech^2 u (sech^2 u - 2 tanh^2 u)
#     M_*^2        = M2_ini exp[-(c_M/tau) sech^2 u]
#
# built from x2 = exp(-2|u|) exactly as gravity_models_hill_valley_smg does, so that they
# stay exact where cosh u would overflow.
# --------------------------------------------------------------------------------------
def _hill_valley_pieces(a, c_M, tau, a_t, M2_ini):
    """``alpha_M``, ``dalpha_M/dln a`` and ``M_*^2`` for the hill/valley parametrisation."""
    u = 0.5 * tau * np.log(a / a_t)
    x2 = np.exp(-2. * np.abs(u))
    opx2 = 1. + x2
    sech2 = 4. * x2 / opx2**2
    tanh_u = np.sign(u) * (1. - x2) / opx2
    alpha_M = c_M * tanh_u * sech2
    dalpha_M = c_M * (0.5 * tau) * sech2 * (sech2 - 2. * tanh_u**2)
    M2 = M2_ini * np.exp(-(c_M / tau) * sech2)
    return alpha_M, dalpha_M, M2


def _rph_hill_valley(a, c_M, tau, a_t, r, M2_ini, alpha_K):
    """
    ``(alpha_K, alpha_M, dalpha_M/da, alpha_B_hi, dalpha_B_hi/da, alpha_T, alpha_T', alpha_T'', M2)``.

    ``alpha_K`` is a genuine constant here -- ``parameters_smg[0]``, mapped by
    ``cosmoprimo`` onto ``RPHkineticitymodel=1`` (constant) -- so it passes straight
    through.  Contrast :func:`_rph_propto_omega`.
    """
    alpha_M, dalpha_M_dlna, M2 = _hill_valley_pieces(a, c_M, tau, a_t, M2_ini)
    z = np.zeros_like(alpha_M + a)
    return (alpha_K + z, alpha_M, dalpha_M_dlna / a,
            -r * alpha_M, -r * dalpha_M_dlna / a,
            z, z, z, M2)


def _rph_propto_omega(a, lna, bg, c_k, c_b, c_m, c_t, M2_ini):
    r"""
    Same tuple for ``propto_omega``, :math:`\alpha_i = c_i \Omega_{\rm smg}(a)`.

    HEFTCAMB's ``Ode = grhov/(3 H^2)`` is mochi_class' ``Omega_smg`` (``007p2_RPH.f90``
    line 1362), so the shape function is shared with :mod:`mochiclass_stability`.
    :math:`M_\ast^2` comes from the same cumulative trapezoid of
    :math:`\int \alpha_M\,d\ln a`; ``RPHintegratefromtoday=False`` makes ``M2_ini`` the
    early-time value on both sides.

    Converting log- to ``a``-derivatives: :math:`d/da = a^{-1} d/d\ln a` and
    :math:`d^2/da^2 = a^{-2}(d^2/d\ln a^2 - d/d\ln a)`.
    """
    Om, dOm, d2Om = bg['Omega_de'], bg['dOmega_de'], bg['d2Omega_de']
    dl = np.diff(lna, axis=-1)
    integ = np.concatenate([np.zeros(Om.shape[:-1] + (1,)),
                            np.cumsum(0.5 * (Om[..., 1:] + Om[..., :-1]) * dl, axis=-1)], axis=-1)
    M2 = M2_ini * np.exp(c_m * integ)
    # alpha_K runs too: c_k is RPHkineticity_ODE0, and alpha_K(a) = c_k Omega_smg(a).
    # It does not matter to mochi_class (alpha_K cancels out of the gradient test) but it
    # is the leading term of EFT_kinetic at early times, where a constant alpha_K would
    # make the kinetic term diverge like adotoa^2 instead of vanishing.
    return (c_k * Om, c_m * Om, c_m * dOm / a,
            c_b * Om, c_b * dOm / a,
            c_t * Om, c_t * dOm / a, c_t * (d2Om - dOm) / a**2,
            M2)


# --------------------------------------------------------------------------------------
# EFT functions
# --------------------------------------------------------------------------------------
def eft_functions(a, bg, h, alpha_K, alpha_M, dalpha_M, alpha_B_hi, dalpha_B_hi,
                  alpha_T, dalpha_T, d2alpha_T, M2):
    # NOTE alpha_K is a function of a in general (propto_omega), not the input constant.
    """
    The EFT-basis quantities EFTCAMB's stability module reads, from the RPH alphas.

    ``alpha_B_hi`` is in the hi_class convention and is converted here; ``dalpha_*`` are
    ``d/da``.  Returns a dict with the pieces :func:`eft_kinetic` and :func:`eft_gradient`
    need, plus ``EFTAT`` / ``EFTDT`` / ``one_plus_Omega`` for the tensor and
    Newton-constant tests, and ``tol`` for the relative comparison.
    """
    h0 = h / _C_OVER_100
    E2, P_tot = bg['E2'], bg['P_tot']
    ad = a * h0 * np.sqrt(E2)                       # adotoa = a H / c
    Hdot = ad**2 * (1. - 1.5 * (1. + P_tot))        # d(adotoa)/d(conformal time)
    grhov = 3. * h0**2 * bg['Omega_de'] * E2 * a**2  # 8 pi G rho_de a^2
    w = bg['w']

    AB = BRAIDING_EFTCAMB_OVER_HICLASS * alpha_B_hi
    AB_P = BRAIDING_EFTCAMB_OVER_HICLASS * dalpha_B_hi

    Mp1 = M2
    PM_V = M2 - 1.
    PM_P = alpha_M * Mp1 / a
    PM_PP = ((-alpha_M + alpha_M**2 + a * dalpha_M) * Mp1) / a**2

    OmV = PM_V + alpha_T * Mp1
    OmP = PM_P + alpha_T * PM_P + dalpha_T * Mp1
    OmPP = PM_PP + alpha_T * PM_PP + 2. * dalpha_T * PM_P + d2alpha_T * Mp1

    c = ((ad**2 - Hdot) * (OmV + 0.5 * a * OmP) - 0.5 * (a * ad)**2 * OmPP
         + 0.5 * grhov * (1. + w))
    G1V = 0.25 * (alpha_K * Mp1 * ad**2 - 2. * c) / (h0**2 * a**2)
    G2V = (2. * AB * Mp1 - a * OmP) * ad / (h0 * a)
    G2P = (-(2. * AB * Mp1 - a * OmP) * ad / (h0 * a**2)
           - (-2. * Mp1 * (AB_P * ad**2 + AB * Hdot / a) - 2. * AB * ad**2 * PM_P
              + OmP * (ad**2 + Hdot) + a * ad**2 * OmPP) / (h0 * a * ad))
    G3V = -alpha_T * Mp1
    G3P = -PM_P * alpha_T - Mp1 * dalpha_T
    G4V, G4P = -G3V, -G3P
    G5V, G5P = 0.5 * G3V, 0.5 * G3P

    one_plus_Omega = 1. + OmV
    return {'a': a, 'h0': h0, 'adotoa': ad, 'Hdot': Hdot, 'EFTc': c,
            # alpha_T = 0 kills Gamma3, Gamma4 and Gamma5 outright, which removes most of
            # the gradient polynomial.  Recorded here so eft_gradient can take the short
            # route; the two branches are algebraically identical.
            'alpha_T_zero': not np.any(alpha_T),
            'OmegaV': OmV, 'OmegaP': OmP, 'OmegaPP': OmPP,
            'Gamma1V': G1V, 'Gamma2V': G2V, 'Gamma2P': G2P,
            'Gamma4V': G4V, 'Gamma4P': G4P, 'Gamma5V': G5V, 'Gamma5P': G5P,
            'one_plus_Omega': one_plus_Omega,
            'EFTAT': one_plus_Omega - G4V, 'EFTDT': one_plus_Omega,
            'tol': STABILITY_THRESHOLD * ad**2 * one_plus_Omega**2}


def eft_kinetic(f):
    """
    ``EFT_kinetic``, transcribed from ``06_abstract_EFTCAMB_model.f90`` line 555.

    Reproduces HEFTCAMB's own array to ~1e-6 relative on a single model, 5.5e-5 worst case
    over a broad prior.  Note that ``EFTc`` cancels
    identically between the ``4 c (1+Omega-Gamma4)`` term and ``8 Gamma1 (1+Omega-Gamma4)``
    -- which is why this one quantity survives the ``get_eft_functions`` ordering bug
    described in the module docstring.

    The ``6 adotoa Gamma2 OmegaP`` cross term takes one power of ``h0``, not two.  Upstream
    it sat inside the inner bracket, which left it dimensionally inconsistent with its
    neighbours and suppressed by ``h0 ~ 2e-4`` -- effectively dropped.  Placed as below, and
    with ``alpha_T = 0`` for definiteness, the expression collapses exactly to

    .. code::

        EFT_kinetic = 18 M2^3 adotoa^2 ( alpha_K + 3/2 alpha_B^2 ) = 18 M2^3 adotoa^2 D

    the standard Horndeski no-ghost quantity, carrying the same ``18 M2^3 adotoa^2``
    prefactor as ``EFT_gradient = 18 M2^3 adotoa^2 cs2num``.  With the stray ``h0`` it does
    not.  Checked to 1e-12 relative against :func:`mochiclass_stability.kinetic_D`; the
    matching HEFTCAMB fix is in ``heftcamb_upstream_fixes.patch``.

    This changes no verdict for ``alpha_K >= 0``: both forms are ``2 alpha_K`` plus a sum of
    squares, so neither can go negative there and the ghost test cannot fire either way.  It
    matters for the *reported* value (wrong by up to 48%), and for ``alpha_K < 0``.
    """
    a, ad, h0 = f['a'], f['adotoa'], f['h0']
    opo = f['one_plus_Omega'] - f['Gamma4V']
    return 9. * opo * (4. * f['EFTc'] * opo + 3. * ad**2 * f['OmegaP']**2 * a**2
                       + a**2 * h0 * (h0 * (3. * f['Gamma2V']**2 + 8. * f['Gamma1V'] * opo)
                                      + 6. * ad * f['Gamma2V'] * f['OmegaP']))


def eft_gradient(f):
    """
    ``EFT_gradient``, transcribed from ``06_abstract_EFTCAMB_model.f90`` line 559.

    Cannot be validated against ``get_eft_functions`` -- that dump evaluates it with
    ``EFTc = 0`` (see the module docstring) -- so it is validated through the verdict
    instead, in ``validate_heftcamb.py``.

    When ``alpha_T`` vanishes identically (always, for hill/valley) every ``Gamma4`` and
    ``Gamma5`` term drops and the polynomial collapses to four terms.  That branch is
    taken automatically; ``validate_heftcamb.py`` checks the two against each other.
    """
    a, ad, h0, Hdot, c = f['a'], f['adotoa'], f['h0'], f['Hdot'], f['EFTc']
    if f.get('alpha_T_zero'):
        OV, OP = f['OmegaV'], f['OmegaP']
        G2V, G2P = f['Gamma2V'], f['Gamma2P']
        opo = 1. + OV
        return 9. * (3. * a**2 * ad**2 * OP**2 * opo - a**2 * G2V**2 * h0**2 * opo
                     + 4. * c * opo**2
                     - 2. * a * ad * h0 * (a * G2P * opo**2 + G2V * opo * (opo - a * OP)))
    OV, OP, OPP = f['OmegaV'], f['OmegaP'], f['OmegaPP']
    G2V, G2P = f['Gamma2V'], f['Gamma2P']
    G4V, G4P, G5V, G5P = f['Gamma4V'], f['Gamma4P'], f['Gamma5V'], f['Gamma5P']
    return 9. * (
        8. * a * ad**2 * G5P - 16. * ad**2 * G5V**2 + 16. * c * G5V**2
        - 2. * a**2 * ad**2 * G4P * OP + 4. * a**2 * ad**2 * G5P * OP
        - 4. * a * ad**2 * G5V * OP - 4. * a**2 * ad**2 * G4P * G5V * OP
        - 8. * a * ad**2 * G5V**2 * OP + 3. * a**2 * ad**2 * OP**2
        + 4. * a**2 * ad**2 * G5V * OP**2 + 16. * a * ad**2 * G5P * OV
        + 16. * c * G5V * OV - 16. * ad**2 * G5V**2 * OV
        - 2. * a**2 * ad**2 * G4P * OP * OV + 4. * a**2 * ad**2 * G5P * OP * OV
        - 4. * a * ad**2 * G5V * OP * OV + 3. * a**2 * ad**2 * OP**2 * OV
        + 8. * a * ad**2 * G5P * OV**2
        - a**2 * G2V**2 * h0**2 * (1. + OV)
        + 4. * a**2 * ad**2 * G5V * OPP * (1. + 2. * G5V + OV)
        + 4. * c * (4. * G5V + (1. + OV)**2)
        - 2. * a * ad * h0 * (a * G2P * (1. - G4V + OV) * (1. + 2. * G5V + OV)
                              + G2V * (G4V * (-1. + 2. * a * G5P + 2. * G5V + a * OP - OV)
                                       + (1. + OV) * (1. + a * (G4P - 2. * G5P - OP) + OV)
                                       - 2. * G5V * (1. - a * G4P + a * OP + OV)))
        + 8. * G5V * Hdot + 16. * G5V**2 * Hdot + 4. * a * G5V * OP * Hdot
        + 8. * a * G5V**2 * OP * Hdot + 16. * G5V * OV * Hdot
        + 16. * G5V**2 * OV * Hdot + 4. * a * G5V * OP * OV * Hdot
        + 8. * G5V * OV**2 * Hdot
        + 4. * G4V**2 * (ad**2 * (1. + 2. * a * G5P + 4. * G5V + a * OP + OV)
                         - (1. + 2. * G5V + OV) * Hdot)
        + 2. * G4V * (ad**2 * (-(a**2 * OP**2) + a**2 * OPP * (1. + 2. * G5V + OV)
                               - 4. * (1. + OV) * (1. + 2. * a * G5P + 4. * G5V + OV)
                               - a * OP * (3. + 2. * a * G5P + 2. * G5V + 3. * OV))
                      + (1. + 2. * G5V + OV) * (4. + a * OP + 4. * OV) * Hdot))


# --------------------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------------------
_COSMO_KEYS = mcs._COSMO_KEYS
_CHUNK_ELEMENTS = mcs._CHUNK_ELEMENTS

#: Stability flags this module does *not* reproduce.  All are off in ``cosmoprimo``'s
#: ``heftcamb`` engine; :func:`stable` refuses to answer if you say one is on.
_UNSUPPORTED_FLAGS = ('EFT_ghost_math_stability', 'EFT_mass_math_stability',
                      'EFT_mass_stability', 'EFT_additional_priors',
                      'EFT_positivity_bounds', 'EFT_minkowski_limit')


def _check_flags(flags):
    if not flags:
        return
    on = sorted(k for k in _UNSUPPORTED_FLAGS if flags.get(k))
    if on:
        raise ValueError(
            'heftcamb_stability reproduces only the ghost and gradient conditions '
            '(EFT_ghost_stability / EFT_gradient_stability), which is what cosmoprimo\'s '
            'heftcamb engine switches on. It cannot predict a run with {} enabled.'
            .format(', '.join(on)))


def _refined_grid(a_grid, a_t, tau, a_start, u_max=8., n_refine=96):
    """Per-model window around the hill/valley transition; see the mochiclass twin."""
    u = np.linspace(-u_max, u_max, int(n_refine))
    a_ref = np.clip(a_t[:, None] * np.exp(2. * u[None, :] / tau[:, None]), a_start, 1.)
    return np.concatenate([np.broadcast_to(a_grid, (a_t.size, a_grid.size)), a_ref], axis=-1)


def _scan(model, par, cosmo_keys, a_grid, refine, n_refine, chunk, a_start):
    """One sweep of the ``a`` grid, returning everything EFTCAMB reduces over it."""
    n = len(next(iter(par.values())))
    lna = np.log(a_grid)
    width = a_grid.size + (n_refine if refine else 0)
    nchunk = chunk or max(1, _CHUNK_ELEMENTS // width)

    keys = ('min_kinetic', 'min_gradient', 'min_AT', 'min_DT', 'min_one_plus_Omega')
    out = {k: np.empty(n) for k in keys}
    for i in range(0, n, nchunk):
        sl = slice(i, min(i + nchunk, n))
        p = {k: v[sl][:, None] for k, v in par.items()}
        bgkw = {k: p[k] for k in cosmo_keys}
        if model == 'hill_valley':
            a = (_refined_grid(a_grid, par['a_t'][sl], par['tau'][sl], a_start,
                               n_refine=n_refine) if refine else a_grid)
            bg = background(a, derivs=False, **bgkw)
            rph = _rph_hill_valley(a, p['c_M'], p['tau'], p['a_t'], p['r'], p['M2_ini'],
                                   p['alpha_K'])
        else:
            a = a_grid
            bg = background(a, derivs=True, **bgkw)
            rph = _rph_propto_omega(a, lna, bg, p['alpha_K'], p['c_b'], p['c_m'],
                                    p['c_t'], p['M2_ini'])
        f = eft_functions(a, bg, p['h'], *rph)
        tol = f['tol']
        # EFTCAMB tests "< -tol" pointwise, and tol depends on a, so the margin has to be
        # minimised, not the bare term.
        out['min_kinetic'][sl] = (eft_kinetic(f) + tol).min(axis=-1)
        out['min_gradient'][sl] = (eft_gradient(f) + tol).min(axis=-1)
        shape = f['adotoa'].shape
        out['min_AT'][sl] = np.broadcast_to(f['EFTAT'], shape).min(axis=-1)
        out['min_DT'][sl] = np.broadcast_to(f['EFTDT'], shape).min(axis=-1)
        out['min_one_plus_Omega'][sl] = np.broadcast_to(f['one_plus_Omega'], shape).min(axis=-1)
    return out


def _prepare(args, cosmo, na, a_grid, a_start):
    bad = set(cosmo) - set(_COSMO_KEYS)
    if bad:
        raise TypeError('unexpected argument(s) {}; expected model parameters or one of {}'
                        .format(sorted(bad), list(_COSMO_KEYS)))
    if 'h' not in cosmo:
        cosmo = dict(cosmo, h=0.6736)
    names = list(args) + list(cosmo)
    par, shape, n = mcs._broadcast(names, list(args.values()) + list(cosmo.values()))
    grid = default_a_grid(na, a_start) if a_grid is None else np.asarray(a_grid, dtype='f8')
    return par, shape, n, grid, list(cosmo)


def scan_hill_valley(c_M, tau, a_t, r=2., M2_ini=1., alpha_K=1e-4, na=DEFAULT_NA,
                     a_grid=None, a_start=A_START, refine=True, n_refine=96, chunk=None,
                     **cosmo):
    """
    Everything EFTCAMB minimises over ``a``, for the hill/valley parametrisation.

    Returns a dict of arrays: ``min_kinetic`` and ``min_gradient`` are the *margins*
    (term + tolerance, so ``>= 0`` passes), ``min_AT`` / ``min_DT`` /
    ``min_one_plus_Omega`` are the bare minima.  ``alpha_K`` is a real argument here,
    unlike in :mod:`mochiclass_stability`: it enters ``EFT_kinetic`` through ``Gamma1``.
    """
    args = dict(c_M=c_M, tau=tau, a_t=a_t, r=r, M2_ini=M2_ini, alpha_K=alpha_K)
    par, shape, n, grid, ck = _prepare(args, cosmo, na, a_grid, a_start)
    out = _scan('hill_valley', par, ck, grid, refine, n_refine, chunk, a_start)
    return {k: (v.reshape(shape) if shape else v[0]) for k, v in out.items()}


def scan_propto_omega(c_b, c_m, c_t=0., M2_ini=1., alpha_K=1., na=DEFAULT_NA,
                      a_grid=None, a_start=A_START, chunk=None, **cosmo):
    """Same, for ``propto_omega``.  The grid must stay sorted (``M_*^2`` is a running integral)."""
    args = dict(c_b=c_b, c_m=c_m, c_t=c_t, M2_ini=M2_ini, alpha_K=alpha_K)
    par, shape, n, grid, ck = _prepare(args, cosmo, na, a_grid, a_start)
    out = _scan('propto_omega', par, ck, grid, False, 0, chunk, a_start)
    return {k: (v.reshape(shape) if shape else v[0]) for k, v in out.items()}


def _verdict(s):
    """
    EFTCAMB's verdict from a scan, with ``cosmoprimo``'s default flags.

    ``EFT_ghost_stability`` gives the kinetic and tensor-ghost tests plus the positive
    Newton constant; ``EFT_gradient_stability`` gives the gradient and tensor-gradient
    tests plus the same Newton-constant test.  Both are on by default.
    """
    ok = s['min_kinetic'] >= 0.
    ok = np.logical_and(ok, s['min_gradient'] >= 0.)
    ok = np.logical_and(ok, s['min_AT'] >= -STABILITY_THRESHOLD)
    ok = np.logical_and(ok, s['min_DT'] >= -STABILITY_THRESHOLD)
    ok = np.logical_and(ok, s['min_one_plus_Omega'] > 0.)
    return ok


def stable_hill_valley(c_M, tau, a_t, r=2., M2_ini=1., alpha_K=1e-4, flags=None, **kwargs):
    """``True`` where HEFTCAMB would run the model, ``False`` where it would abort."""
    _check_flags(flags)
    return _verdict(scan_hill_valley(c_M, tau, a_t, r=r, M2_ini=M2_ini,
                                     alpha_K=alpha_K, **kwargs))


def stable_propto_omega(c_b, c_m, c_t=0., M2_ini=1., alpha_K=1., flags=None, **kwargs):
    """``True`` where HEFTCAMB would run the model, ``False`` where it would abort."""
    _check_flags(flags)
    return _verdict(scan_propto_omega(c_b, c_m, c_t=c_t, M2_ini=M2_ini,
                                      alpha_K=alpha_K, **kwargs))


def stable(gravity_model, parameters_smg, flags=None, **cosmo):
    """
    Dispatch on mochi_class' ``gravity_model`` / ``parameters_smg`` pair.

    Deliberately the same signature as :func:`mochiclass_stability.stable`, since
    ``cosmoprimo``'s ``heftcamb`` engine takes the same arguments as its ``mochiclass``
    one -- so the two gates are drop-in swaps for each other.
    """
    p = np.asarray(parameters_smg, dtype='f8')
    if gravity_model in ('hill_valley', 'no_slip_gravity'):
        if p.shape[-1] != 6:
            raise ValueError('hill_valley expects parameters_smg = [alpha_K, c_M, tau, '
                             'a_t, r, M2_ini] on the last axis, got {}'.format(p.shape))
        aK, c_M, tau, a_t, r, M2_ini = (p[..., i] for i in range(6))
        return stable_hill_valley(c_M, tau, a_t, r=r, M2_ini=M2_ini, alpha_K=aK,
                                  flags=flags, **cosmo)
    if gravity_model == 'propto_omega':
        if p.shape[-1] != 5:
            raise ValueError('propto_omega expects parameters_smg = [alpha_K, c_b, c_m, '
                             'c_t, M2_ini] on the last axis, got {}'.format(p.shape))
        aK, c_b, c_m, c_t, M2_ini = (p[..., i] for i in range(5))
        return stable_propto_omega(c_b, c_m, c_t=c_t, M2_ini=M2_ini, alpha_K=aK,
                                   flags=flags, **cosmo)
    raise ValueError("gravity_model must be 'hill_valley' ('no_slip_gravity') or "
                     "'propto_omega', got {!r}".format(gravity_model))
