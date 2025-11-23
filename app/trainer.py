"""trainer.py
Coordinates the variational and exact simulations. The main loop updates
parameters using McLachlan's rule, evaluates the ansatz state and compares it
to the exact reference trajectory.
"""

from __future__ import annotations

import numpy as np

from .ansatz import ansatz_state, prepare_initial_state
from .hamiltonian import initial_state
from .simulator_exact import run_exact_sim
from .vqs_core import vqs_update


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """Compute fidelity between two pure states."""

    return float(np.abs(np.vdot(psi, phi)) ** 2)


def run_vqs(initial_params: np.ndarray, t_span: float, dt: float) -> tuple:
    """Run the variational simulation.

    Args:
        initial_params: Starting parameter vector.
        t_span: Final time value.
        dt: Time step size.

    Returns:
        Tuple of (state_history, param_history) arrays.
    """

    num_steps = int(t_span / dt) + 1
    param_history = np.zeros((num_steps, len(initial_params)), dtype=float)
    state_history = np.zeros((num_steps, 2), dtype=complex)

    params = initial_params.copy()
    param_history[0] = params
    state_history[0] = ansatz_state(params)

    for k in range(1, num_steps):
        t = (k - 1) * dt
        params = vqs_update(params, t, dt)
        param_history[k] = params
        state_history[k] = ansatz_state(params)

    return state_history, param_history


def run_full_simulation(t_span: float = 5.0, dt: float = 0.05):
    """Execute variational and exact simulations and compare trajectories."""

    params0 = prepare_initial_state()
    exact_init = initial_state()

    var_states, param_hist = run_vqs(params0, t_span, dt)
    exact_states = run_exact_sim(exact_init, t_span, dt)

    fidelities = []
    for psi_var, psi_exact in zip(var_states, exact_states):
        fidelities.append(fidelity(psi_var, psi_exact))
    fidelities = np.array(fidelities)

    times = np.linspace(0.0, t_span, int(t_span / dt) + 1)
    return {
        "times": times,
        "variational_states": var_states,
        "exact_states": exact_states,
        "fidelities": fidelities,
        "param_history": param_hist,
    }


__all__ = ["run_full_simulation", "run_vqs", "fidelity"]
