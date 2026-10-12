#!/usr/bin/env python3
"""Compare time integration algorithms in a minimal molecular-dynamics system.

A diatomic molecule is represented by two identical atoms connected by a
harmonic bond. Its motion is microcanonical (NVE): there is no thermostat,
friction, or external energy source. A correct time integrator should therefore
keep the total energy approximately constant over the trajectory.

Algorithms:
    1. Forward Euler       -- simple, but unstable for oscillatory motion
    2. Symplectic Euler    -- bounded energy oscillations, relatively inaccurate
    3. Velocity Verlet     -- standard, time-reversible symplectic MD integrator
    4. Runge-Kutta 4 (RK4) -- accurate at small timesteps, not symplectic

Outputs in --outdir:
    integrator_comparison.png
    timestep_sensitivity.png
    long_time_energy.png
    integrator_trajectories.csv
    integration_summary.csv

Run:
    pip install numpy matplotlib
    python md_integrator_comparison.py
    python md_integrator_comparison.py --dt 0.3 --duration 40 --outdir results

All quantities are in *reduced units*, not real nm/ps/eV. This is a teaching
example, not a realistic force field. NumPy and Matplotlib are the only deps.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works over SSH and in batch jobs without a display.
import matplotlib.pyplot as plt
import numpy as np

# Two-atom, one-dimensional harmonic bond: V = k/2 * (x2 - x1 - r0)^2.
MASS = 1.0               # mass of each atom (reduced units)
SPRING_CONSTANT = 1.0    # k (reduced units)
EQUILIBRIUM_BOND = 1.0   # r0 (reduced units)
INITIAL_EXTENSION = 0.30  # q = x2 - x1 - r0
INITIAL_REL_VELOCITY = 0.12  # qdot = v2 - v1

METHODS = ["Forward Euler", "Symplectic Euler", "Velocity Verlet", "RK4"]
STYLES = {
    "Forward Euler":    {"color": "tab:red",    "linestyle": "--"},
    "Symplectic Euler": {"color": "tab:orange", "linestyle": "-"},
    "Velocity Verlet":  {"color": "tab:blue",   "linestyle": "-"},
    "RK4":              {"color": "tab:green",  "linestyle": "-"},
}


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray       # shape [n_times, 2]
    v: np.ndarray       # shape [n_times, 2]
    extension: np.ndarray
    relative_velocity: np.ndarray
    energy: np.ndarray
    force_calls: int


def initial_state() -> tuple[np.ndarray, np.ndarray]:
    """Choose a stationary center of mass, but an excited bond vibration."""
    separation = EQUILIBRIUM_BOND + INITIAL_EXTENSION
    x = np.array([-separation / 2.0, separation / 2.0], dtype=float)
    v = np.array([-INITIAL_REL_VELOCITY / 2.0,
                   INITIAL_REL_VELOCITY / 2.0], dtype=float)
    return x, v


def bond_extension(x: np.ndarray) -> np.ndarray:
    return x[..., 1] - x[..., 0] - EQUILIBRIUM_BOND


def total_energy(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    kinetic = 0.5 * MASS * np.sum(v**2, axis=-1)
    potential = 0.5 * SPRING_CONSTANT * bond_extension(x)**2
    return kinetic + potential


def exact_solution(t: np.ndarray) -> np.ndarray:
    """Analytic relative displacement for two equal masses on a spring."""
    omega = np.sqrt(2.0 * SPRING_CONSTANT / MASS)
    return (INITIAL_EXTENSION * np.cos(omega * t)
            + (INITIAL_REL_VELOCITY / omega) * np.sin(omega * t))


def simulate(method: str, dt: float, duration: float) -> Trajectory:
    """Integrate Newton's equations; count acceleration evaluations."""
    if method not in METHODS:
        raise ValueError(f"Unknown integration method: {method}")
    if dt <= 0 or duration <= 0:
        raise ValueError("The timestep and duration must both be positive")

    n = max(1, int(round(duration / dt)))
    t = np.arange(n + 1, dtype=float) * dt
    xs = np.empty((n + 1, 2))
    vs = np.empty((n + 1, 2))
    x, v = initial_state()
    xs[0], vs[0] = x, v
    force_calls = 0

    def acceleration(positions: np.ndarray) -> np.ndarray:
        nonlocal force_calls
        force_calls += 1
        # If the bond is stretched, atom 1 feels +F and atom 2 feels -F.
        force = SPRING_CONSTANT * bond_extension(positions)
        return np.array([force, -force]) / MASS

    # Velocity Verlet retains the previous acceleration, requiring only one
    # *new* force evaluation per timestep after the first one.
    a = acceleration(x) if method == "Velocity Verlet" else None

    for i in range(n):
        if method == "Forward Euler":
            # Both updates use the OLD state. This is NOT an energy-stable MD
            # algorithm for a vibrating bond.
            old_a = acceleration(x)
            next_x = x + dt * v
            next_v = v + dt * old_a

        elif method == "Symplectic Euler":
            # Kick first, then drift using the NEW velocity.
            next_v = v + dt * acceleration(x)
            next_x = x + dt * next_v

        elif method == "Velocity Verlet":
            # Half-kick / drift / half-kick, algebraically rearranged.
            next_x = x + dt * v + 0.5 * dt**2 * a
            next_a = acceleration(next_x)
            next_v = v + 0.5 * dt * (a + next_a)
            a = next_a

        else:  # RK4 applied to the combined state (positions + velocities)
            k1x, k1v = v, acceleration(x)
            k2x = v + 0.5 * dt * k1v
            k2v = acceleration(x + 0.5 * dt * k1x)
            k3x = v + 0.5 * dt * k2v
            k3v = acceleration(x + 0.5 * dt * k2x)
            k4x = v + dt * k3v
            k4v = acceleration(x + dt * k3x)
            next_x = x + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            next_v = v + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

        x, v = next_x, next_v
        xs[i + 1], vs[i + 1] = x, v

    return Trajectory(
        t=t, x=xs, v=vs, extension=bond_extension(xs),
        relative_velocity=vs[:, 1] - vs[:, 0],
        energy=total_energy(xs, vs), force_calls=force_calls,
    )


def make_comparison_figure(runs: dict[str, Trajectory], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    reference = runs["Velocity Verlet"]
    e0 = reference.energy[0]
    omega = np.sqrt(2 * SPRING_CONSTANT / MASS)

    # Short-time bond motion. Euler's amplitude begins growing even though
    # the exact isolated molecule keeps vibrating at constant amplitude.
    ax = axes[0, 0]
    ax.plot(reference.t, exact_solution(reference.t), color="black",
            linewidth=2, label="Exact solution")
    for method, traj in runs.items():
        ax.plot(traj.t, traj.extension, label=method, **STYLES[method], alpha=0.9)
    ax.set_xlim(0, min(13.0, reference.t[-1]))
    ax.set_ylim(-1.2, 1.2)  # Deliberately clips large, unphysical Euler motion.
    ax.set_title("Bond vibration: Euler becomes unphysical")
    ax.set_xlabel("Time (reduced units)")
    ax.set_ylabel(r"Bond extension $q = x_2-x_1-r_0$")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.25)

    # Energy on a log axis makes the explosive Euler failure obvious.
    ax = axes[0, 1]
    for method, traj in runs.items():
        ax.plot(traj.t, traj.energy / e0, label=method, **STYLES[method])
    ax.axhline(1, color="black", linestyle=":", linewidth=1.4,
               label="Exact NVE energy")
    ax.set_yscale("log")
    ax.set_xlabel("Time (reduced units)")
    ax.set_ylabel(r"Normalized total energy $E(t)/E(0)$")
    ax.set_title("Total energy: NVE must not heat itself")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left")

    # Zoom in, excluding Euler, to compare stable algorithms fairly.
    ax = axes[1, 0]
    for method in METHODS[1:]:
        traj = runs[method]
        ax.plot(traj.t, traj.energy / e0, label=method, **STYLES[method])
    ax.axhline(1, color="black", linestyle=":", linewidth=1.4)
    ax.set_xlabel("Time (reduced units)")
    ax.set_ylabel(r"Normalized total energy $E(t)/E(0)$")
    ax.set_title("Zoom: bounded error vs. slow numerical damping")
    ax.set_ylim(0.65, 1.35)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # Phase space: conservative oscillation follows a closed elliptical orbit.
    ax = axes[1, 1]
    theta = np.linspace(0, 2 * np.pi, 500)
    amplitude = np.hypot(INITIAL_EXTENSION, INITIAL_REL_VELOCITY / omega)
    ax.plot(amplitude * np.cos(theta), amplitude * np.sin(theta),
            color="black", linewidth=2.3, label="Exact closed orbit")
    for method, traj in runs.items():
        selection = traj.t <= 12.0
        ax.plot(traj.extension[selection],
                traj.relative_velocity[selection] / omega,
                label=method, **STYLES[method], alpha=0.85)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Phase space: Euler spirals outward")
    ax.set_xlabel("Bond extension q")
    ax.set_ylabel(r"Relative velocity $\dot{q}/\omega$")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    fig.suptitle("Molecular dynamics: numerical integration of a vibrating diatomic bond",
                 fontsize=15, fontweight="bold")
    fig.savefig(outdir / "integrator_comparison.png", dpi=190)
    plt.close(fig)


def make_timestep_figure(duration: float, outdir: Path) -> None:
    # Energy error across a range of timesteps. The same undamped molecule is
    # simulated for approximately the same physical duration each time.
    dts = [0.05, 0.10, 0.20, 0.30, 0.45, 0.65, 0.85]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for method in METHODS:
        errors = []
        for dt in dts:
            run = simulate(method, dt, duration)
            errors.append(max(1e-15, float(np.max(np.abs(
                run.energy / run.energy[0] - 1.0)))))
        ax.plot(dts, errors, marker="o", linewidth=2, label=method,
                **STYLES[method])
    ax.set_yscale("log")
    ax.set_xlabel("Integration timestep Δt (reduced units)")
    ax.set_ylabel("Maximum relative total-energy error")
    ax.set_title("Smaller timesteps help, but integrator choice also matters")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.savefig(outdir / "timestep_sensitivity.png", dpi=190)
    plt.close(fig)


def make_long_time_figure(outdir: Path) -> None:
    """Show why velocity Verlet is preferred even over accurate, non-symplectic RK4."""
    dt = 0.40
    duration = 500.0
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    for method in ("Velocity Verlet", "RK4"):
        run = simulate(method, dt, duration)
        ax.plot(run.t, run.energy / run.energy[0], label=method,
                linewidth=1.5, **STYLES[method])
    ax.axhline(1, color="black", linestyle=":", linewidth=1.5,
               label="Exact NVE energy")
    ax.set_xlim(0, duration)
    ax.set_ylim(0.45, 1.15)
    ax.set_title("Long runs: velocity Verlet conserves energy without systematic drift")
    ax.set_xlabel("Time (reduced units)")
    ax.set_ylabel(r"Normalized total energy $E(t)/E(0)$")
    ax.text(0.03, 0.08, "Both methods: Δt = 0.40; duration = 500\n"
            "Velocity Verlet: bounded fluctuations\n"
            "RK4: artificial energy loss over long times",
            transform=ax.transAxes, fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "lightgray", "alpha": 0.92})
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.savefig(outdir / "long_time_energy.png", dpi=190)
    plt.close(fig)


def save_csv(runs: dict[str, Trajectory], outdir: Path) -> None:
    # CSV holds bond displacements and energies, enough for student replots.
    t = runs["Velocity Verlet"].t
    energy0 = runs["Velocity Verlet"].energy[0]
    with (outdir / "integrator_trajectories.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "exact_extension"] + [
            f"{m.lower().replace(' ', '_')}_{field}"
            for m in METHODS for field in ("extension", "energy_ratio")
        ])
        for i, time in enumerate(t):
            values = [float(time), float(exact_solution(np.array([time]))[0])]
            for m in METHODS:
                values += [float(runs[m].extension[i]),
                           float(runs[m].energy[i] / energy0)]
            writer.writerow(values)

    with (outdir / "integration_summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "final_energy_ratio", "max_relative_energy_error",
                         "max_absolute_bond_error", "acceleration_evaluations"])
        for m in METHODS:
            run = runs[m]
            writer.writerow([
                m, float(run.energy[-1] / energy0),
                float(np.max(np.abs(run.energy / energy0 - 1))),
                float(np.max(np.abs(run.extension - exact_solution(t)))),
                run.force_calls,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dt", type=float, default=0.20,
                        help="MD timestep in reduced units (default: 0.20)")
    parser.add_argument("--duration", type=float, default=45.0,
                        help="Simulation duration (default: 45)")
    parser.add_argument("--outdir", type=Path, default=Path("md_integrator_results"),
                        help="Directory to save PNG and CSV outputs")
    args = parser.parse_args()
    if args.dt <= 0 or args.duration <= 0:
        parser.error("--dt and --duration must be positive")

    args.outdir.mkdir(parents=True, exist_ok=True)
    runs = {m: simulate(m, args.dt, args.duration) for m in METHODS}
    save_csv(runs, args.outdir)
    make_comparison_figure(runs, args.outdir)
    make_timestep_figure(args.duration, args.outdir)
    make_long_time_figure(args.outdir)

    omega = np.sqrt(2.0 * SPRING_CONSTANT / MASS)
    print(f"Diatomic harmonic bond; NVE; dt = {args.dt:g}; duration = {args.duration:g}")
    print(f"Vibrational angular frequency = {omega:.4f} (reduced units)")
    print(f"Velocity Verlet stability requires dt < 2/omega = {2/omega:.4f}")
    print("-" * 95)
    print(f"{'Method':<19} {'Final E/E0':>13} {'Max |ΔE/E0|':>16} "
          f"{'Max |q-q_exact|':>19} {'Force calls':>14}")
    for method in METHODS:
        run = runs[method]
        print(f"{method:<19} {run.energy[-1]/run.energy[0]:>13.4g} "
              f"{np.max(np.abs(run.energy/run.energy[0]-1)):>16.4g} "
              f"{np.max(np.abs(run.extension-exact_solution(run.t))):>19.4g} "
              f"{run.force_calls:>14}")
    print("-" * 95)
    print("Velocity Verlet should maintain small, bounded energy oscillations.")
    print("RK4 may be very accurate here, but uses ~4 forces/step and is not symplectic.")
    print("All schemes can become inaccurate if the timestep is excessively large.")
    print(f"Figures and CSV data saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
