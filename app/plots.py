"""plots.py
Utility functions for generating SVG plots used in the examples folder. All
functions are lightweight and rely on matplotlib with explicit `format="svg"`
so that the repository contains only text-based vector graphics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_exact_vs_variational(times: np.ndarray, fidelities: np.ndarray, output_path: str) -> None:
    """Plot fidelity between exact and variational trajectories over time."""

    plt.figure(figsize=(6, 4))
    plt.plot(times, fidelities, label="Fidelity |⟨ψ_var|ψ_exact⟩|²", color="purple")
    plt.xlabel("Time")
    plt.ylabel("Fidelity")
    plt.ylim(0, 1.05)
    plt.title("Exact vs Variational Agreement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_parameter_evolution(times: np.ndarray, param_history: np.ndarray, output_path: str) -> None:
    """Plot how each parameter changes over time."""

    plt.figure(figsize=(6, 4))
    for idx in range(param_history.shape[1]):
        plt.plot(times, param_history[:, idx], label=fr"θ{idx + 1}")
    plt.xlabel("Time")
    plt.ylabel("Parameter value (rad)")
    plt.title("Variational Parameter Evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_bloch_trajectory(states: np.ndarray, output_path: str) -> None:
    """Plot the path of the state on the Bloch sphere (projected to xy and z)."""

    # Convert statevector [a, b] to Bloch coordinates
    bloch_vectors = []
    for a, b in states:
        sx = 2 * np.real(np.conj(a) * b)
        sy = 2 * np.imag(np.conj(b) * a)
        sz = np.abs(a) ** 2 - np.abs(b) ** 2
        bloch_vectors.append([sx, sy, sz])
    bloch_vectors = np.array(bloch_vectors)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(bloch_vectors[:, 0], bloch_vectors[:, 1], bloch_vectors[:, 2], color="teal", lw=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("State Trajectory on Bloch Sphere")
    # Draw sphere outline for context
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


__all__ = [
    "plot_exact_vs_variational",
    "plot_parameter_evolution",
    "plot_bloch_trajectory",
]
