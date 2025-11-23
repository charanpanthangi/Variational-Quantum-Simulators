"""main.py
Command-line entry point for running the variational quantum simulator.
The script performs both variational and exact simulations, saves SVG figures
and prints simple summary metrics so beginners can follow what happened.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .plots import (
    plot_bloch_trajectory,
    plot_exact_vs_variational,
    plot_parameter_evolution,
)
from .trainer import run_full_simulation


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run a tiny variational quantum simulator")
    parser.add_argument("--tmax", type=float, default=5.0, help="Final time value")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_full_simulation(t_span=args.tmax, dt=args.dt)

    os.makedirs("examples", exist_ok=True)
    plot_exact_vs_variational(
        results["times"], results["fidelities"], "examples/vqs_exact_vs_variational.svg"
    )
    plot_parameter_evolution(
        results["times"], results["param_history"], "examples/vqs_parameter_evolution.svg"
    )
    plot_bloch_trajectory(results["variational_states"], "examples/vqs_state_trajectory_bloch.svg")

    avg_fid = np.mean(results["fidelities"])
    min_fid = np.min(results["fidelities"])
    print("Variational Quantum Simulation complete!")
    print(f"Time grid: 0 → {args.tmax} with dt={args.dt}")
    print(f"Average fidelity vs exact: {avg_fid:.4f}")
    print(f"Worst-case fidelity: {min_fid:.4f}")
    print("SVG plots saved in examples/:")
    print(" - vqs_exact_vs_variational.svg")
    print(" - vqs_parameter_evolution.svg")
    print(" - vqs_state_trajectory_bloch.svg")


if __name__ == "__main__":
    main()
