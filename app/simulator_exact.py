"""simulator_exact.py
Classical reference simulator using matrix exponentials. The goal is to provide
an exact trajectory so that the variational approximation can be compared
quantitatively.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .hamiltonian import hamiltonian_matrix


def exact_step(state: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Apply one exact time-evolution step using exp(-i H dt).

    Args:
        state: Current statevector.
        t: Current time value.
        dt: Time step size.

    Returns:
        Updated statevector after applying the unitary.
    """

    H = hamiltonian_matrix(t)
    unitary = expm(-1j * H * dt)
    return unitary @ state


def run_exact_sim(initial_state: np.ndarray, t_span: float, dt: float) -> np.ndarray:
    """Run exact simulation over a time grid.

    Args:
        initial_state: Starting state vector.
        t_span: Final time value.
        dt: Time step size.

    Returns:
        Array of shape (num_steps, state_dim) with the full trajectory.
    """

    num_steps = int(t_span / dt) + 1
    states = np.zeros((num_steps, initial_state.shape[0]), dtype=complex)
    states[0] = initial_state
    current = initial_state
    for k in range(1, num_steps):
        t = (k - 1) * dt
        current = exact_step(current, t, dt)
        # normalize to reduce numerical drift
        current = current / np.linalg.norm(current)
        states[k] = current
    return states


__all__ = ["exact_step", "run_exact_sim"]
