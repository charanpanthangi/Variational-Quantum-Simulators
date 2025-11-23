"""Tests for the variational ansatz."""

import numpy as np

from app.ansatz import ansatz_state, derivative_states, prepare_initial_state


def test_state_is_normalized():
    params = prepare_initial_state()
    state = ansatz_state(params)
    assert np.isclose(np.linalg.norm(state), 1.0)


def test_derivative_shape():
    params = prepare_initial_state()
    derivs = derivative_states(params)
    assert derivs.shape == (len(params), 2)
