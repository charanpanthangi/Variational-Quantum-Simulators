# Variational Quantum Simulator (VQS)

## What This Project Does
- Simulates quantum dynamics using a variational quantum circuit.
- Tracks how parameters evolve according to a variational update rule.
- Compares approximate variational dynamics to exact matrix exponential evolution.

## Why VQS Is Interesting
- Classical simulation of dynamics is expensive; variational methods reduce cost.
- Works on near-term devices (NISQ).
- Teaches how PQCs can approximate physical processes.

## Why SVG Instead of PNG
> CODEX cannot preview PNG/JPG and displays
> “Binary files are not supported.”
> All images here are SVG to ensure safe rendering and diff-friendly behavior.

## How It Works
- Define time-dependent Hamiltonian.
- Construct parameterized ansatz.
- Apply McLachlan variational update rule.
- Integrate parameters over time.
- Compare with exact evolution.

## Repository Layout
- `app/` core Python modules (Hamiltonian, ansatz, VQS update rule, plotting).
- `examples/` contains saved SVG figures from a sample run.
- `notebooks/` interactive tutorial using PennyLane.
- `tests/` lightweight pytest checks for key pieces.

## How to Run
```bash
pip install -r requirements.txt
python app/main.py --tmax 5 --dt 0.05
```

### Expected Output
- SVG plots in `examples/` folder:
  - fidelity curve
  - parameter evolution
  - Bloch trajectory

## Future Work
- Multi-qubit VQS
- Adaptive ansatz
- Real hardware experiments
