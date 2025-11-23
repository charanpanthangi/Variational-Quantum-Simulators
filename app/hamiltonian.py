"""hamiltonian.py
This module defines a simple time-dependent Hamiltonian for a single qubit.
The Hamiltonian used throughout the examples is
    H(t) = a(t) * Z + b(t) * X
with smooth coefficients a(t) = cos(t), b(t) = sin(t).
The helper functions return both a PennyLane Hamiltonian object and a
matrix representation for exact simulations.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def a_coeff(t: float) -> float:
    """Coefficient multiplying the Z term.

    Args:
        t: Time value.

    Returns:
        Cosine envelope for the Z term.
    """

    return float(np.cos(t))


def b_coeff(t: float) -> float:
    """Coefficient multiplying the X term.

    Args:
        t: Time value.

    Returns:
        Sine envelope for the X term.
    """

    return float(np.sin(t))


def H_of_t(t: float) -> qml.Hamiltonian:
    """Construct the time-dependent PennyLane Hamiltonian H(t).

    Args:
        t: Time value.

    Returns:
        qml.Hamiltonian combining PauliZ and PauliX with smooth coefficients.
    """

    coeffs = [a_coeff(t), b_coeff(t)]
    ops = [qml.PauliZ(0), qml.PauliX(0)]
    return qml.Hamiltonian(coeffs, ops)


def hamiltonian_matrix(t: float) -> np.ndarray:
    """Return the 2x2 matrix representation of H(t).

    Args:
        t: Time value.

    Returns:
        Complex-valued matrix suitable for exact simulation.
    """

    a_t = a_coeff(t)
    b_t = b_coeff(t)
    z_mat = np.array([[1.0, 0.0], [0.0, -1.0]])
    x_mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    return a_t * z_mat + b_t * x_mat


def initial_state() -> np.ndarray:
    """Return the |0> computational basis state vector.

    The state is used as the starting point for both the variational
    simulation and the exact reference evolution.
    """

    return np.array([1.0 + 0.0j, 0.0 + 0.0j])


__all__ = ["H_of_t", "hamiltonian_matrix", "initial_state", "a_coeff", "b_coeff"]
