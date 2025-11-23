"""vqs_core.py
Implements McLachlan's variational principle for updating ansatz parameters.
The rule enforces that the time derivative of the variational state stays as
close as possible to the exact Schrödinger evolution.

Mathematically, we solve A(t) * theta_dot = C(t) where
    A_ij = Re(<dpsi_i | dpsi_j>)
    C_i  = Im(<dpsi_i | H | psi>)
The update step is explicit Euler for simplicity.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .ansatz import ansatz_state, derivative_states
from .hamiltonian import H_of_t


def compute_A_matrix(params: np.ndarray) -> np.ndarray:
    """Compute the quantum geometric tensor matrix A.

    Args:
        params: Current parameter vector.

    Returns:
        Real-valued matrix with shape (p, p).
    """

    derivs = derivative_states(params)
    num_params = derivs.shape[0]
    A = np.zeros((num_params, num_params), dtype=float)
    for i in range(num_params):
        for j in range(num_params):
            overlap = np.vdot(derivs[i], derivs[j])
            A[i, j] = np.real(overlap)
    # Add tiny diagonal shift to avoid singular matrices in early steps
    A += 1e-6 * np.eye(num_params)
    return A


def compute_C_vector(params: np.ndarray, t: float) -> np.ndarray:
    """Compute the right-hand side vector C for the VQS equation.

    Args:
        params: Current parameter vector.
        t: Current time.

    Returns:
        Real vector with length equal to number of parameters.
    """

    psi = ansatz_state(params)
    derivs = derivative_states(params)
    H = qml.matrix(H_of_t(t))()
    C = []
    for dpsi in derivs:
        element = np.vdot(dpsi, H @ psi)
        C.append(np.imag(element))
    return np.array(C, dtype=float)


def vqs_update(params: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Perform one Euler update for the parameter vector using McLachlan's rule.

    Args:
        params: Current parameters.
        t: Current time.
        dt: Time step.

    Returns:
        Updated parameters after one step.
    """

    A = compute_A_matrix(params)
    C = compute_C_vector(params, t)
    theta_dot = np.linalg.solve(A, C)
    return params + dt * theta_dot


__all__ = ["compute_A_matrix", "compute_C_vector", "vqs_update"]
