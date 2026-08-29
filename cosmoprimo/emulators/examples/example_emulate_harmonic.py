"""Emulating the harmonic section -- CMB Cl -- end to end.

Four steps, and the expensive one is separate on purpose::

    space = Space(samples=chain)                  # where it must be accurate
    emu   = Emulator(cosmo, space)                # build -- cheap, nothing runs yet
    emu.train(budget=2, checkpoint=..., chunk=..) # pay -- hours of CAMB, resumable
    emu.write('harmonic.h5')                       # keep

and then, in the likelihood, it is just another engine::

    Cosmology(engine='harmonic.h5').clone(**point).get_harmonic().lensed_cl()

Run:  python example_emulate_harmonic.py    (a couple of minutes at this budget and ellmax)
"""

import logging

import numpy as np

from cosmoprimo import Cosmology
from cosmoprimo.emulators import Emulator, emulate, Space

# training progress goes to a logger, not to stdout: without this the run is silent
logging.basicConfig(level=logging.INFO)


# --- 0. the fiducial ---------------------------------------------------------------------------
# Every training node is computed with this cosmology's engine and settings, so set `lensing`,
# `ellmax_cl` and any precision parameters here. Nothing else gets a say later.
fiducial = Cosmology(engine='camb', lensing=True, ellmax_cl=2500)


# --- 1. where the emulator must be accurate ----------------------------------------------------
# This is the single largest lever there is. A chain is worth orders of magnitude more than
# ranges, because the grid then sits on the posterior's principal axes instead of in a rectangle
# around them -- measured 350x in the median and 3600x at the 90th percentile, at equal node
# count. A rectangle around a thin ellipsoid is mostly volume the chain never visits, and the
# interpolant spends its resolution there.
#
#     space = Space(samples=chain)                       # best: mean, covariance and support
#     space = Space(mean=best_fit, covariance=fisher, params=names)   # a Fisher matrix
#
# Standing in for a chain here, with Planck-like correlations (Omega_m and h at -0.92):
names = ['Omega_m', 'Omega_b', 'h', 'logA', 'n_s', 'tau_reio']
mean = np.array([0.3153, 0.0493, 0.6736, 3.045, 0.9649, 0.0544])
sigma = np.array([0.0073, 0.0009, 0.0054, 0.014, 0.0042, 0.0073])
correlation = np.array([[1.00, 0.35, -0.92, 0.28, -0.35, 0.20],
                        [0.35, 1.00, -0.45, 0.10, -0.30, 0.05],
                        [-0.92, -0.45, 1.00, -0.25, 0.44, -0.18],
                        [0.28, 0.10, -0.25, 1.00, 0.15, 0.92],
                        [-0.35, -0.30, 0.44, 0.15, 1.00, 0.10],
                        [0.20, 0.05, -0.18, 0.92, 0.10, 1.00]])
chain = np.random.default_rng(42).multivariate_normal(
    mean, correlation * np.outer(sigma, sigma), size=20000)
space = Space(samples={name: chain[:, index] for index, name in enumerate(names)})
print(space, '\n  correlated:', space.is_correlated(), '-> the grid will be whitened')


# --- 2. build ----------------------------------------------------------------------------------
# `Emulator` does not train. Training is hours of Boltzmann calls, so it is a deliberate,
# separate, resumable step -- which is what lets you size it first, below. When a run is small
# enough not to need that, `emulate(...)` builds and trains in one call.
#
# `basis='physical'` trains in (omega_cdm, omega_b, h) while the space stays in (Omega_m, ...):
# the spectra respond simply to the physical densities, and Omega_m -> omega_cdm mixes in h, so
# whitening -- a rotation and a rescale -- cannot absorb it. Measured 1.5x in the median and
# 3.2x at the 90th percentile on a space given as samples. Do not turn it on for a space given
# as plain limits: mapping a box gives the bounding box of a curved image, and that came out
# 5.7x worse.
emu = Emulator(fiducial, space,
              section='harmonic',
              of=('lensed_cl', 'lens_potential_cl'),   # add 'unlensed_cl' if a likelihood wants it
              basis='physical')

print('\nexpands  :', emu.params)
print('exact    :', emu.exact_params, '-- handled in code, costing no nodes and unbounded')
for budget in (1, 2, 3):
    print(f'  budget {budget}: {len(emu.nodes(budget=budget))} nodes')

# The per-axis `level` and the total `budget` are different knobs. A level sets one axis's own
# error -- raising one from 2 to 3 cut that axis 276x for 4 extra nodes -- while the budget buys
# only interaction terms. Both are nested, so raising either later reuses every evaluation
# already made. Size the run before paying for it.


# --- 3. train ----------------------------------------------------------------------------------
# For a real run always pass `checkpoint` and `chunk`: a kill then costs one node, not the
# training, and rerunning continues where it stopped.
#
#     emu.train(budget=3, checkpoint='cl_nodes.npz', chunk='30min')   # rerun until complete
#
# `batch_size=N` calls the calculator with dicts of arrays of N nodes instead of one node at a
# time, when it can batch; `mpicomm=...` splits the nodes across ranks.
emu.train(budget=1)


# --- 4. check it, before believing it ----------------------------------------------------------
# Leads with sigma, not the mean: under importance reweighting a constant offset cancels exactly
# and only the scatter costs effective sample size. The default metric is
# rms(difference) / rms(reference), a ratio of norms -- never a pointwise ratio, which would
# divide by zero wherever TE crosses zero. Pass a chi2 against a real covariance when you have
# one; that is the number that actually matters.
report = emu.validate(npoints=12, seed=7)
print('\n', report, sep='')


# --- 5. use it ---------------------------------------------------------------------------------
point = {name: float(value) for name, value in zip(names, mean)}

predicted = emu.predict(**point)                       # a dict of arrays
print('\npredicted:', sorted(predicted)[:3], '...')

# ... or get the thing you started with back: a Cosmology, engine and all. A dict of arrays is a
# training artifact; what a likelihood wants is a cosmology.
fast = emu.to_cosmology()
table = fast.clone(**point).get_harmonic().lensed_cl()
truth = fiducial.clone(**point).get_harmonic().lensed_cl()
good = truth['ell'] >= 30
error = np.abs(table['tt'][good] / truth['tt'][good] - 1.)
print(f'lensed TT above l=30: median |dCl/Cl| = {np.median(error):.2e}, '
      f'worst = {error.max():.2e}')

# Outside the trained box it raises rather than extrapolating. That is deliberate: one clipped
# draw measured dchi2 2e4 where every draw inside stayed below 0.2. Silent clipping turns an
# obvious failure into a plausible wrong answer.
from cosmoprimo.emulators import CoverageError

try:
    emu.predict(**{**point, 'Omega_m': 0.5})
except CoverageError as exc:
    print('\noutside the box:', str(exc).split(' Extrapolation')[0])


# --- 6. keep it --------------------------------------------------------------------------------
# HDF5, because a trained emulator outlives the session that made it: readable by anything,
# browsable with `h5ls -r`, and it does not execute code when opened. A `version` token travels
# with it and a mismatched read fails loudly rather than quietly returning nonsense.
path = emu.write('harmonic_emulator.h5')
print('\nsaved to', path)

reloaded = Cosmology(engine=path)                      # like engine='camb', but instant
again = reloaded.clone(**point).get_harmonic().lensed_cl()
assert np.allclose(again['tt'], table['tt'], rtol=1e-12)
print("Cosmology(engine='harmonic_emulator.h5') reproduces it exactly")


# --- what to reach for next --------------------------------------------------------------------
#
# Several sections at once share one Boltzmann call per node, which is the entire cost. If a
# likelihood needs distances or a sound horizon as well as Cl, ask for them together rather than
# training twice over the same grid:
#
#     Emulator(fiducial, space, section=['harmonic', 'thermodynamics'])
#     Emulator(fiducial, space, section={'harmonic': dict(of=('lensed_cl',)),
#                                        'fourier':  dict(z=np.linspace(0., 3., 20))})
#
# For many more parameters than this, the sparse grid stops being payable and the MLP engine
# takes over -- same interface, one word:
#
#     emu.train(engine='mlp', nsamples=8192)
#
# It is a stochastic fit, not an interpolant, so it does not converge the way the grid does: on a
# 2-parameter CMB space it reached 1.9e-3 from 128 nodes where the grid reached 8e-7 from 13.
# Reach for it when the dimension is the problem, not the accuracy.
