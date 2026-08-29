"""Tests for the resumable training.

Encodes the operational failures that actually cost work: a training that could not resume, a
node set whose infeasibility surfaced only at fit time, and a fit fed a partial node set.
"""

import numpy as np
import pytest

from cosmoprimo.emulators.tools.training import Training, NodeEvaluationError, _seconds


def target(params):
    a, b = params['a'], params['b']
    return {'y': np.array([a, b, a * b])}


NODES = np.array([[value, 2. * value] for value in np.linspace(0., 1., 10)])


def test_seconds_parsing():
    assert _seconds('30min') == 1800.
    assert _seconds('2h') == 7200.
    assert _seconds('45s') == 45.
    assert _seconds(120) == 120.
    assert _seconds(None) is None


def test_resume_reproduces_the_uninterrupted_result(tmp_path):
    """A training that cannot resume loses everything to one kill."""
    checkpoint = str(tmp_path / 'nodes.npz')
    whole = Training(target, NODES, ['a', 'b'])
    whole.run()

    # a budget shorter than one evaluation: each run must still advance by exactly one node,
    # or resumption never terminates
    partial = Training(target, NODES, ['a', 'b'], checkpoint=checkpoint, chunk=1e-9, save_every=1)
    assert not partial.run()
    assert partial.done == 1
    while not (resumed := Training(target, NODES, ['a', 'b'], checkpoint=checkpoint,
                                   chunk=1e-9, save_every=1)).run():
        pass
    assert resumed.done == len(NODES)
    assert np.allclose(np.array(resumed.values['y']), np.array(whole.values['y']))
    assert np.allclose(np.array(resumed.keys), np.array(whole.keys))


def test_a_failing_node_is_reported_not_swallowed():
    def flaky(params):
        if params['a'] > 0.5:
            raise RuntimeError('solver did not converge')
        return target(params)

    training = Training(flaky, NODES, ['a', 'b'])
    with pytest.raises(NodeEvaluationError, match='did not converge'):
        training.run()


def test_an_incomplete_node_set_is_refused():
    """A sparse-grid fit needs every node of its combination; a partial set must not look fine."""
    training = Training(target, NODES, ['a', 'b'], chunk=1e-9, save_every=1)
    training.run()
    assert training.done == 1 and not training.complete
    with pytest.raises(ValueError, match='incomplete'):
        training.inputs()


def test_outputs_round_trip():
    training = Training(target, NODES, ['a', 'b'])
    training.run()
    assert np.allclose(training.inputs()[:, 0], NODES[:, 0])
    assert np.allclose(training.outputs()['y'][:, 2], NODES[:, 0] * NODES[:, 1])


def test_the_target_applies_its_own_transform():
    """No `forward` hook: whatever the target returns is what gets fitted."""
    def scaled(params):
        return {name: value / params['a'] for name, value in target(params).items()}

    training = Training(scaled, NODES[1:], ['a', 'b'])
    training.run()
    stored = np.array(training.values['y'])
    assert np.allclose(stored[:, 0], 1.)          # y[0] = a, divided by a


#: The same function as `target`, taking a whole batch at once. Records the batch sizes it was
#: handed, which is what the two tests below are actually about.
BATCH_SIZES = []


def batched(params):
    BATCH_SIZES.append(np.size(params['a']))
    return {'y': np.stack([params['a'], params['b'], params['a'] * params['b']], axis=-1)}


def run(fn, **kwargs):
    training = Training(fn, NODES, ['a', 'b'], **kwargs)
    training.run()
    return training


@pytest.mark.parametrize('batch_size', [len(NODES), 3, 2])
def test_batching_is_an_optimisation_not_a_different_answer(batch_size):
    """Whatever the batch size -- including one that does not divide the node set -- the fitted
    values must be the ones the node-at-a-time path produces."""
    BATCH_SIZES.clear()
    many, one = run(batched, batch_size=batch_size), run(target)
    assert np.allclose(np.array(many.outputs()['y']), np.array(one.outputs()['y']))
    assert np.allclose(many.inputs(), one.inputs())
    # padding must not leak into the training data
    assert len(many.outputs()['y']) == len(NODES)


def test_every_batch_has_exactly_the_same_size():
    """A jitted target must see one shape: a ragged last batch would retrace the whole pipeline
    for a handful of nodes. The tail is padded and the extra results dropped."""
    BATCH_SIZES.clear()
    assert len(NODES) % 3 != 0        # so without padding the last batch would be short
    run(batched, batch_size=3)
    assert set(BATCH_SIZES) == {3}
    # ... and a size is a size: batches are at most `batch_size`, not all of them at once
    BATCH_SIZES.clear()
    run(batched, batch_size=2)
    assert set(BATCH_SIZES) == {2}


def test_a_batch_that_returns_the_wrong_length_is_caught():
    """Silently taking the first row would train the emulator on the wrong function."""
    def wrong(params):
        return {'y': np.ones(3)}          # one node's worth, whatever the batch

    with pytest.raises(NodeEvaluationError, match='leading node axis'):
        Training(wrong, NODES, ['a', 'b'], batch_size=2).run()


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match='at least 1'):
        Training(lambda params: {'y': np.ones(3)}, NODES, ['a', 'b'], batch_size=0)
