"""Tests for the validation report.

The first two encode the interpretation errors that this session actually made: reading a mean
offset as a cost, and folding box-coverage failures into the interpolation-error distribution.
"""

import numpy as np
import pytest

from cosmoprimo.emulators.tools.space import Space
from cosmoprimo.emulators.tools.validation import Validation, validate


def test_a_constant_offset_costs_nothing():
    """mean 3.4 with sigma 0 is a perfect proposal: the offset cancels in the weights."""
    report = Validation(np.full(50, 3.4))
    assert report.offset == pytest.approx(3.4)
    assert report.sigma == pytest.approx(0., abs=1e-12)
    assert report.ess == pytest.approx(1., abs=1e-9)


def test_scatter_is_what_degrades_ess():
    tight = Validation(np.random.default_rng(1).normal(3.4, 0.21, size=4000))
    loose = Validation(np.random.default_rng(1).normal(0.0, 4.0, size=4000))
    assert tight.ess > 0.98        # large mean, small scatter -> essentially free
    assert loose.ess < 0.05        # zero mean, large scatter -> unusable
    assert loose.offset == pytest.approx(0., abs=0.2)


def test_coverage_failures_are_separated_not_averaged_in():
    """One clipped point gave dchi2 2e4 where every in-box point was below 0.2; averaging them
    together makes both numbers meaningless."""
    space = Space(bounds={'a': (0., 1.)})
    points = [{'a': 0.5}, {'a': 0.9}, {'a': 5.0}]      # the last is outside
    report = validate(predict=lambda params: {'y': np.array([params['a']])},
                      truth=lambda params: {'y': np.array([params['a']])},
                      points=points, space=space,
                      metric=lambda p, t: float(np.sum((p['y'] - t['y'])**2)))
    assert report.coverage_failures == 1
    assert report.errors.size == 2
    assert report.npoints == 3
    assert report.coverage == pytest.approx(2. / 3.)
    assert 'COVERAGE' in report.summary()


def test_proxy_metric_says_so():
    plain = Validation(np.ones(5), metric='cosmic_variance', proxy=False)
    proxy = Validation(np.ones(5), metric='cosmic_variance', proxy=True)
    assert 'PROXY' not in plain.summary()
    assert 'PROXY' in proxy.summary()


def test_validate_reports_coverage_and_scatter_together():
    """Both numbers matter and neither substitutes for the other: a coverage failure is fixed by
    resizing the box, a large scatter by adding nodes."""
    space = Space(bounds={'a': (0., 1.)})
    points = [{'a': value} for value in (0.2, 0.4, 0.6, 5.0)]
    report = validate(predict=lambda params: {'y': np.array([params['a'] + 0.1])},
                      truth=lambda params: {'y': np.array([params['a']])},
                      points=points, space=space,
                      metric=lambda p, t: float(np.sum((p['y'] - t['y'])**2)))
    assert report.coverage_failures == 1
    assert report.sigma == pytest.approx(0., abs=1e-12)   # a constant offset: no scatter
    assert report.offset == pytest.approx(0.01, rel=1e-6)
