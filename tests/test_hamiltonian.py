"""Tests for time-dependent Hamiltonian construction."""

import numpy as np

from app.hamiltonian import H_of_t, hamiltonian_matrix


def test_hamiltonian_matches_matrix():
    t = 0.3
    H_pl = H_of_t(t)
    H_mat = hamiltonian_matrix(t)
    # Compare PennyLane matrix with manual matrix
    pl_mat = np.array(H_pl.matrix())
    assert np.allclose(pl_mat, H_mat)
    # Check Hermiticity
    assert np.allclose(pl_mat.conj().T, pl_mat)
