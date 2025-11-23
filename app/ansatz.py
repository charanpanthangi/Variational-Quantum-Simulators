"""ansatz.py
Defines a lightweight parameterized quantum circuit (PQC) for one qubit.
The circuit applies three rotations RX, RY and RZ. Because the repository
focuses on pedagogy, all steps are commented and a helper function returns
the statevector so it can be compared with exact dynamics.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

# Single-qubit device used for all demonstrations
DEV = qml.device("default.qubit", wires=1)


def prepare_initial_state() -> np.ndarray:
    """Prepare the starting parameters for the ansatz.

    Returns:
        Small numpy array with three angles (in radians).
    """

    # Start near |0> with tiny rotations so the update rule has room to move.
    return np.array([0.05, 0.05, 0.05], dtype=float)


@qml.qnode(DEV)
def ansatz_state(params: np.ndarray) -> np.ndarray:
    """Return the statevector produced by the parameterized circuit.

    Args:
        params: Array with three rotation angles.

    Returns:
        Complex statevector of length 2 (for one qubit).
    """

    rx, ry, rz = params
    qml.RX(rx, wires=0)
    qml.RY(ry, wires=0)
    qml.RZ(rz, wires=0)
    return qml.state()


def ansatz_expectations(params: np.ndarray) -> float:
    """Compute expectation value of Z for quick diagnostics.

    This function is intentionally simple and keeps run-times short.
    """

    state = ansatz_state(params)
    # <Z> = |0|^2 - |1|^2 for one-qubit state |psi> = [a, b]
    return float(np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2)


def derivative_states(params: np.ndarray) -> np.ndarray:
    """Return derivative of the statevector with respect to each parameter.

    Args:
        params: Current parameter array.

    Returns:
        Array with shape (num_params, state_dim) containing d|psi>/dθ_i.
    """

    # autograd supports jacobian on complex outputs; PennyLane exposes it via qml.jacobian
    jac_fn = qml.jacobian(ansatz_state, argnum=0)
    jac = jac_fn(params)
    return np.array(jac)


__all__ = ["ansatz_state", "ansatz_expectations", "derivative_states", "prepare_initial_state", "DEV"]
