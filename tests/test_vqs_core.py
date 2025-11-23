"""Tests for McLachlan core update routines."""

import numpy as np

from app.ansatz import prepare_initial_state
from app.vqs_core import compute_A_matrix, compute_C_vector, vqs_update


def test_shapes_and_finiteness():
    params = prepare_initial_state()
    A = compute_A_matrix(params)
    C = compute_C_vector(params, t=0.0)
    assert A.shape == (len(params), len(params))
    assert C.shape == (len(params),)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(C))


def test_vqs_update_moves_parameters():
    params = prepare_initial_state()
    new_params = vqs_update(params, t=0.0, dt=0.1)
    assert new_params.shape == params.shape
    assert not np.allclose(new_params, params)
