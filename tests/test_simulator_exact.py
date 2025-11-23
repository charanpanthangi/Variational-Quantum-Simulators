"""Tests for the exact simulator."""

import numpy as np

from app.hamiltonian import initial_state
from app.simulator_exact import exact_step


def test_exact_step_changes_state():
    psi0 = initial_state()
    psi1 = exact_step(psi0, t=0.0, dt=0.1)
    # State should remain normalized but differ from start
    assert np.isclose(np.linalg.norm(psi1), 1.0)
    assert not np.allclose(psi1, psi0)
