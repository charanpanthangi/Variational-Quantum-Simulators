"""End-to-end check for the training loop."""

import numpy as np

from app.trainer import run_full_simulation


def test_full_simulation_runs():
    results = run_full_simulation(t_span=0.2, dt=0.1)
    assert "fidelities" in results
    assert results["fidelities"].shape[0] == int(0.2 / 0.1) + 1
    assert np.all(results["fidelities"] <= 1.0)
    assert np.all(results["fidelities"] >= 0.0)
