"""Measuring how good an emulator is -- reporting the number that actually matters.

Two lessons, both learned the expensive way, are built into the report:

**1. The scatter is the cost, not the mean.**  Under importance reweighting or smc a constant
offset in dchi2 cancels exactly in the normalised weights; only the point-to-point scatter
degrades the effective sample size, as ``ESS ~ exp(-(sigma/2)^2)``. Measured on a real CMB
comparison: mean dchi2 of 3.4 with sigma 0.21 -- which reads alarming and costs 1.1% of the
sample. Reporting the mean as though it were the cost led to several wrong conclusions before
the scatter was computed.

**2. A point outside the trained box is a different failure.**  It is catastrophic rather than
gradual (measured: dchi2 2e4 for one clipped draw, against a maximum of 0.17 for every draw
inside) and it is fixed by resizing the box, not by adding nodes. Mixing the two into one
distribution makes both uninterpretable, so they are counted separately.

A third guard: a metric may declare itself a proxy. The cosmic-variance dchi2 used throughout
this work is not an upper bound on a real likelihood -- it treats TT/TE/EE as independent while
the true covariance couples them, and measured ~6x optimistic in the median against CamSpec. A
proxy metric says so in its own report rather than letting the reader assume otherwise.
"""

import numpy as np


class Validation(object):
    """The outcome of validating an emulator at a set of points."""

    def __init__(self, errors, coverage_failures=0, metric='dchi2', proxy=False, npoints=None):
        self.errors = np.asarray(errors, dtype='f8')
        self.coverage_failures = int(coverage_failures)
        self.metric, self.proxy = metric, bool(proxy)
        self.npoints = int(npoints if npoints is not None else
                           self.errors.size + self.coverage_failures)

    # ── the number that matters ────────────────────────────────────────────────
    @property
    def sigma(self):
        """Point-to-point scatter -- the quantity that costs effective sample size."""
        return float(self.errors.std()) if self.errors.size else float('nan')

    @property
    def offset(self):
        """Mean, reported separately because it cancels under reweighting."""
        return float(self.errors.mean()) if self.errors.size else float('nan')

    @property
    def ess(self):
        """Effective-sample-size fraction, ``exp(-(sigma/2)^2)``, for one-shot reweighting.

        Pessimistic for smc, which anneals through intermediate distributions and tolerates far
        more scatter than a single importance step.
        """
        return float(np.exp(-(self.sigma / 2.)**2)) if self.errors.size else float('nan')

    def percentile(self, value):
        return float(np.percentile(self.errors, value)) if self.errors.size else float('nan')

    @property
    def median(self):
        return float(np.median(self.errors)) if self.errors.size else float('nan')

    @property
    def worst(self):
        return float(self.errors.max()) if self.errors.size else float('nan')

    @property
    def coverage(self):
        return 1. - self.coverage_failures / max(self.npoints, 1)

    # ── reporting ──────────────────────────────────────────────────────────────
    def summary(self):
        lines = [f'Validation on {self.npoints} points, metric {self.metric!r}'
                 + (' (PROXY -- not an upper bound on a real likelihood)' if self.proxy else ''),
                 f'  sigma        {self.sigma:12.4g}   <- the cost: ESS ~ {self.ess:.3f}',
                 f'  offset       {self.offset:12.4g}   (cancels under reweighting)',
                 f'  median       {self.median:12.4g}',
                 f'  90th pct     {self.percentile(90):12.4g}',
                 f'  worst        {self.worst:12.4g}']
        if self.coverage_failures:
            lines.append(f'  COVERAGE     {self.coverage_failures} of {self.npoints} points '
                         f'outside the trained box ({self.coverage:.1%} covered) -- these are a '
                         f'box-sizing failure, not interpolation error, and are excluded above')
        else:
            lines.append(f'  coverage     all {self.npoints} points inside the trained box')
        return '\n'.join(lines)

    def __repr__(self):
        return (f'Validation(sigma={self.sigma:.4g}, offset={self.offset:.4g}, '
                f'ess={self.ess:.3f}, coverage_failures={self.coverage_failures})')


def validate(predict, truth, points, metric, space=None, proxy=False, metric_name='dchi2'):
    """Compare an emulator against a reference at ``points``.

    Parameters
    ----------
    predict : callable
        ``predict(params) -> dict``, taking a dict as targets do.
    truth : callable
        ``truth(params) -> dict``; a target itself will do.
    points : list of dict
        In user parameters -- draw them from the posterior, not uniformly in the box: uniform
        draws in a high-dimensional box sit overwhelmingly near its boundary and measure corners
        a chain never visits.
    metric : callable
        ``metric(prediction, reference) -> float``.
    space : Space, default=None
        If given, points outside the box are counted as coverage failures instead of being
        silently clipped and folded into the error distribution.
    """
    errors, failures = [], 0
    for point in points:
        if space is not None and not space.contains(point):
            failures += 1
            continue
        # A refusal is a coverage failure too, not an error of infinite size. The emulator
        # returns NaN for a point it will not answer -- inside the box but off the node cloud is
        # the case `space.contains` cannot see -- and folding that into the error distribution
        # would turn every summary statistic into NaN.
        error = float(metric(predict(dict(point)), truth(dict(point))))
        if not np.isfinite(error):
            failures += 1
            continue
        errors.append(error)
    return Validation(errors, coverage_failures=failures, metric=metric_name, proxy=proxy,
                      npoints=len(points))
