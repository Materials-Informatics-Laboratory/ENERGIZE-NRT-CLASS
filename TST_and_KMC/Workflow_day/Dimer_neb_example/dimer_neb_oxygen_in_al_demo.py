#!/usr/bin/env python3
"""
In-class demo: Dimer saddle searches + NEB pathways for O interstitial in FCC Al
================================================================================


Final figure
------------
LEFT PANEL:
    A local collective-coordinate energy landscape built from the actual
    many-atom Dimer trajectories.  The first two principal displacement
    directions define the plotted coordinates, every Dimer trajectory is
    projected into that same space, and the final saddle candidates are marked.

RIGHT PANEL:
    Conventional NEB energy-versus-reaction-coordinate curves for the
    transitions discovered from the Dimer saddles.  The Dimer saddle energy
    is overlaid on each corresponding NEB path.

Important physics caveat
------------------------
ASE's built-in EMT calculator contains parameters for O, but ASE explicitly
warns that H/C/N/O are not well described by EMT.  Therefore this is an
ALGORITHMIC DEMONSTRATION of Dimer and NEB, not a quantitatively reliable
model of oxygen diffusion in aluminum.

"""

from pathlib import Path
import csv
import shutil

import numpy as np
import matplotlib.pyplot as plt

from ase import Atom
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.geometry import find_mic
from ase.io import read, write
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate, NEB
from ase.optimize import BFGS, FIRE


# =============================================================================
# DEMO SETTINGS
# =============================================================================

A0 = 4.05
SUPERCELL = (2, 2, 2)       # 32 Al + 1 O: intentionally small for class

RELAX_FMAX = 0.03
DIMER_FMAX = 0.04
NEB_FMAX = 0.05

MAX_RELAX_STEPS = 300
MAX_DIMER_STEPS = 500
MAX_NEB_STEPS = 400

DIMER_INITIAL_PUSH = 0.10   # Angstrom
DOWNHILL_PUSH = 0.18        # Angstrom along +/- unstable mode

N_NEB_IMAGES = 7
NEB_SPRING = 0.10

# Initial Dimer search directions.
DIMER_DIRECTIONS = {
    "100":  np.array([1.0,  0.0, 0.0]),
    "110":  np.array([1.0,  1.0, 0.0]),
    "111":  np.array([1.0,  1.0, 1.0]),
    "1m10": np.array([1.0, -1.0, 0.0]),
}

# Local collective-coordinate landscape shown in the LEFT panel.
#
# The two plotted coordinates are obtained from the dominant many-atom
# displacement directions sampled by the successful Dimer searches.
PCA_GRID_N = 33
PCA_PADDING = 0.30
PCA_MIN_HALF_WIDTH = 0.55

OUTDIR = Path("dimer_neb_oxygen_in_al_demo_v4")


# =============================================================================
# HELPERS
# =============================================================================

def attach_emt(atoms):
    atoms.calc = EMT()
    return atoms


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / n


def max_force(atoms):
    f = atoms.get_forces()
    return float(np.max(np.linalg.norm(f, axis=1)))


def relax(atoms, logfile=None, trajectory=None, fmax=RELAX_FMAX):
    atoms = atoms.copy()
    attach_emt(atoms)
    opt = BFGS(atoms, logfile=logfile, trajectory=trajectory)
    opt.run(fmax=fmax, steps=MAX_RELAX_STEPS)
    e = float(atoms.get_potential_energy())
    mf = max_force(atoms)
    return atoms, e, mf


def mic_vector(a, b, cell, pbc):
    """Minimum-image vector b-a."""
    dr = np.asarray(b) - np.asarray(a)
    dr_mic, _ = find_mic(dr, cell, pbc=pbc)
    return np.asarray(dr_mic)


def structure_rmsd(a, b):
    dr = b.positions - a.positions
    dr_mic, _ = find_mic(dr, a.cell, pbc=a.pbc)
    dr_mic = np.asarray(dr_mic)
    return float(np.sqrt(np.mean(np.sum(dr_mic**2, axis=1))))


def oxygen_distance(a, b, o_index):
    dr = mic_vector(
        a.positions[o_index],
        b.positions[o_index],
        a.cell,
        a.pbc,
    )
    return float(np.linalg.norm(dr))


# =============================================================================
# 1. BUILD AND RELAX O INTERSTITIAL
# =============================================================================

def build_starting_minimum():
    print("\n" + "=" * 80)
    print("1. BUILD / RELAX O INTERSTITIAL IN FCC Al")
    print("=" * 80)

    base = bulk("Al", "fcc", a=A0, cubic=True).repeat(SUPERCELL)

    guesses = {
        "octahedral": np.array([0.5, 0.5, 0.5]) * A0,
        "tetrahedral": np.array([0.25, 0.25, 0.25]) * A0,
    }

    candidates = []

    for name, o_pos in guesses.items():
        atoms = base.copy()
        atoms.append(Atom("O", position=o_pos))
        atoms.wrap()
        o_index = len(atoms) - 1

        cdir = OUTDIR / f"candidate_{name}"
        cdir.mkdir(parents=True, exist_ok=True)

        relaxed, e, mf = relax(
            atoms,
            logfile=str(cdir / "relax.log"),
            trajectory=str(cdir / "relax.traj"),
        )
        write(cdir / "relaxed.xyz", relaxed)

        print(
            f"{name:12s}: E = {e: .6f} eV, "
            f"max|F| = {mf:.4f} eV/A"
        )
        candidates.append((e, name, relaxed, o_index))

    candidates.sort(key=lambda x: x[0])
    e0, name, minimum, o_index = candidates[0]

    write(OUTDIR / "initial_minimum.traj", minimum)
    write(OUTDIR / "initial_minimum.xyz", minimum)

    print(f"\nSelected starting minimum: {name}")
    print(f"Energy: {e0:.6f} eV")
    print(f"O atom index: {o_index}")
    print(f"O position: {minimum.positions[o_index]}")

    return minimum, float(e0), o_index


# =============================================================================
# 3. DIMER SEARCHES
# =============================================================================

def run_dimer(initial, initial_energy, o_index, label, direction):
    sdir = OUTDIR / f"dimer_search_{label}"
    sdir.mkdir(parents=True, exist_ok=True)

    work = initial.copy()
    attach_emt(work)

    # Use only the O atom to define the INITIAL mode guess.
    mask = [False] * len(work)
    mask[o_index] = True

    disp = np.zeros((len(work), 3), dtype=float)
    disp[o_index] = DIMER_INITIAL_PUSH * normalize(direction)

    print("\n" + "-" * 80)
    print(f"Dimer search {label}")
    print(f"Initial O direction = {normalize(direction)}")
    print("-" * 80)

    traj_path = sdir / "dimer_search.traj"

    with DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",
        mask=mask,
        logfile=str(sdir / "dimer_control.log"),
    ) as d_control:

        d_atoms = MinModeAtoms(work, d_control)
        d_atoms.displace(displacement_vector=disp)

        with MinModeTranslate(
            d_atoms,
            trajectory=str(traj_path),
            logfile=str(sdir / "dimer_translate.log"),
        ) as optimizer:
            converged = optimizer.run(
                fmax=DIMER_FMAX,
                steps=MAX_DIMER_STEPS,
            )

        saddle = work.copy()
        attach_emt(saddle)
        saddle_energy = float(saddle.get_potential_energy())
        curvature = float(d_atoms.get_curvature())
        eigenmode = np.asarray(d_atoms.get_eigenmode(), dtype=float).copy()

    write(sdir / "saddle.traj", saddle)
    write(sdir / "saddle.xyz", saddle)
    np.save(sdir / "unstable_eigenmode.npy", eigenmode)

    # Read the stored Dimer trajectory so every search can be drawn.
    trajectory = []
    if traj_path.exists():
        try:
            trajectory = read(str(traj_path), ":")
        except Exception:
            trajectory = []

    barrier = saddle_energy - initial_energy

    print(f"Converged: {converged}")
    print(f"Saddle energy: {saddle_energy:.6f} eV")
    print(f"Barrier from initial minimum: {barrier:.6f} eV")
    print(f"Curvature: {curvature:.6e}")

    return {
        "label": label,
        "direction": normalize(direction),
        "saddle": saddle,
        "saddle_energy": saddle_energy,
        "barrier_from_initial": barrier,
        "curvature": curvature,
        "eigenmode": eigenmode,
        "converged": bool(converged),
        "trajectory": trajectory,
    }



# =============================================================================
# COLLECTIVE-COORDINATE PROJECTION FOR THE DIMER LANDSCAPE
# =============================================================================

def configuration_displacement(reference, atoms):
    """
    Return the full many-atom displacement vector relative to reference.

    Minimum-image displacements are used atom-by-atom so ordinary periodic
    wrapping does not create artificial jumps in the collective coordinates.
    """
    dr = atoms.positions - reference.positions
    dr_mic, _ = find_mic(dr, reference.cell, pbc=reference.pbc)
    return np.asarray(dr_mic, dtype=float).reshape(-1)


def fit_collective_modes(minimum, dimer_results):
    """
    Obtain two dominant full-system displacement directions from the Dimer data.

    We intentionally define the basis THROUGH the starting minimum rather than
    mean-centering the data.  Equivalently, this is the eigenbasis of the
    displacement covariance about the initial minimum.  The coordinates
    therefore have a simple interpretation:

        q1 = DeltaR dot e1
        q2 = DeltaR dot e2

    where DeltaR contains the Cartesian displacements of ALL atoms.
    """
    successful = [r for r in dimer_results if r["curvature"] < 0]
    training_results = successful if len(successful) >= 2 else dimer_results

    vectors = []

    for result in training_results:
        frames = list(result["trajectory"])

        # Keep the landscape local and avoid over-weighting hundreds of nearly
        # identical optimizer frames.
        if len(frames) > 30:
            ids = np.linspace(0, len(frames) - 1, 30).astype(int)
            frames = [frames[i] for i in ids]

        for frame in frames:
            vectors.append(configuration_displacement(minimum, frame))

        vectors.append(configuration_displacement(minimum, result["saddle"]))

    if len(vectors) < 2:
        raise RuntimeError(
            "Not enough Dimer configurations were generated to define "
            "a two-dimensional collective-coordinate landscape."
        )

    X = np.asarray(vectors, dtype=float)

    # Remove exact/near-zero rows.
    norms = np.linalg.norm(X, axis=1)
    X = X[norms > 1.0e-8]

    if len(X) < 2:
        raise RuntimeError(
            "The Dimer configurations contain insufficient displacement "
            "information for a 2D projection."
        )

    # SVD without mean-centering: the origin remains the starting minimum.
    _, singular_values, vt = np.linalg.svd(X, full_matrices=False)

    if vt.shape[0] < 2:
        raise RuntimeError("Only one independent collective direction found.")

    modes = vt[:2].copy()

    # Deterministic sign convention: orient each mode so that its largest
    # component is positive.  This only affects how the plot is mirrored.
    for i in range(2):
        j = int(np.argmax(np.abs(modes[i])))
        if modes[i, j] < 0:
            modes[i] *= -1.0

    np.save(OUTDIR / "collective_modes.npy", modes)

    variance_like = singular_values**2
    frac = variance_like[:2] / np.sum(variance_like)

    print("\n" + "=" * 80)
    print("COLLECTIVE-COORDINATE BASIS FROM DIMER TRAJECTORIES")
    print("=" * 80)
    print(
        "Fraction of sampled displacement variance represented by "
        f"mode 1 / mode 2: {frac[0]:.3f} / {frac[1]:.3f}"
    )

    return modes, frac


def project_structure(minimum, atoms, modes):
    """Project one atomic configuration into the two collective coordinates."""
    d = configuration_displacement(minimum, atoms)
    return np.array(
        [
            float(np.dot(d, modes[0])),
            float(np.dot(d, modes[1])),
        ]
    )


def project_dimer_results(minimum, dimer_results, modes):
    """
    Project every stored Dimer trajectory and saddle into collective space.
    """
    projected = {}

    for result in dimer_results:
        coords = [np.array([0.0, 0.0])]

        for frame in result["trajectory"]:
            coords.append(project_structure(minimum, frame, modes))

        saddle_q = project_structure(minimum, result["saddle"], modes)
        coords.append(saddle_q)

        projected[result["label"]] = {
            "trajectory": np.asarray(coords, dtype=float),
            "saddle": saddle_q,
        }

    return projected


def collective_plot_limits(projected, dimer_results):
    """
    Define a LOCAL plotting window from successful Dimer searches only.

    Failed/runaway searches are still projected later, but they are not allowed
    to stretch the axes or define the background energy landscape.
    """
    successful_labels = {
        r["label"] for r in dimer_results if r["curvature"] < 0
    }

    arrays = [
        projected[label]["trajectory"]
        for label in successful_labels
        if label in projected
    ]

    if not arrays:
        arrays = [v["trajectory"] for v in projected.values()]

    q = np.vstack(arrays)
    q = np.vstack([q, np.zeros((1, 2))])

    mins = np.min(q, axis=0)
    maxs = np.max(q, axis=0)
    spans = np.maximum(maxs - mins, 2.0 * PCA_MIN_HALF_WIDTH)

    mins = mins - PCA_PADDING * spans
    maxs = maxs + PCA_PADDING * spans

    # Ensure the starting minimum is comfortably inside the view.
    for j in range(2):
        mins[j] = min(mins[j], -PCA_MIN_HALF_WIDTH)
        maxs[j] = max(maxs[j], +PCA_MIN_HALF_WIDTH)

    return mins, maxs


def build_collective_coordinate_pes(
    minimum,
    minimum_energy,
    modes,
    projected,
    dimer_results,
):
    """
    Evaluate a 2D local PES in the SAME collective coordinates used to display
    the Dimer trajectories.

    At each grid point the entire structure is displaced as

        R(q1,q2) = R0 + q1*e1 + q2*e2

    where e1/e2 are full 3N-dimensional displacement modes.  This is still a
    2D slice through a very high-dimensional PES, but unlike the previous
    O-x/O-y plot, the background and the Dimer trajectories now use the same
    many-atom coordinates.
    """
    print("\n" + "=" * 80)
    print("BUILDING LOCAL COLLECTIVE-COORDINATE ENERGY LANDSCAPE")
    print("=" * 80)

    mins, maxs = collective_plot_limits(
        projected,
        dimer_results,
    )

    q1 = np.linspace(mins[0], maxs[0], PCA_GRID_N)
    q2 = np.linspace(mins[1], maxs[1], PCA_GRID_N)
    Z = np.empty((PCA_GRID_N, PCA_GRID_N), dtype=float)

    mode1 = modes[0].reshape((-1, 3))
    mode2 = modes[1].reshape((-1, 3))

    for iy, y in enumerate(q2):
        for ix, x in enumerate(q1):
            test = minimum.copy()
            test.positions = (
                minimum.positions
                + x * mode1
                + y * mode2
            )
            test.wrap()
            attach_emt(test)
            Z[iy, ix] = (
                float(test.get_potential_energy())
                - minimum_energy
            )

    rows = []
    for iy, y in enumerate(q2):
        for ix, x in enumerate(q1):
            rows.append((x, y, Z[iy, ix]))

    np.savetxt(
        OUTDIR / "collective_coordinate_pes.csv",
        np.asarray(rows),
        delimiter=",",
        header="q1_A,q2_A,relative_energy_eV",
        comments="",
    )

    print(
        "Collective-coordinate PES range: "
        f"{np.nanmin(Z):.4f} to {np.nanmax(Z):.4f} eV"
    )

    return q1, q2, Z, mins, maxs


# =============================================================================
# 4. DISCOVER THE TWO MINIMA CONNECTED BY EACH DIMER SADDLE
# =============================================================================

def relax_both_sides(result, o_index):
    label = result["label"]
    saddle = result["saddle"]
    mode = np.asarray(result["eigenmode"], dtype=float)

    mode_norm = np.linalg.norm(mode)
    if mode_norm == 0:
        raise RuntimeError(f"Search {label}: zero Dimer eigenmode.")
    mode /= mode_norm

    ddir = OUTDIR / f"dimer_search_{label}" / "downhill"
    ddir.mkdir(parents=True, exist_ok=True)

    minima = []

    for sign, name in [(-1.0, "minus"), (+1.0, "plus")]:
        test = saddle.copy()
        test.positions += sign * DOWNHILL_PUSH * mode
        test.wrap()

        relaxed, e, mf = relax(
            test,
            logfile=str(ddir / f"{name}_relax.log"),
            trajectory=str(ddir / f"{name}_relax.traj"),
        )

        write(ddir / f"{name}_minimum.traj", relaxed)
        write(ddir / f"{name}_minimum.xyz", relaxed)
        minima.append((relaxed, float(e), mf))

    a, ea, _ = minima[0]
    b, eb, _ = minima[1]

    od = oxygen_distance(a, b, o_index)
    rmsd = structure_rmsd(a, b)

    # Basic checks for a useful first-order saddle.
    different_minima = (od > 0.20 or rmsd > 0.10)
    below_saddle = (
        result["saddle_energy"] > ea + 1e-4
        and result["saddle_energy"] > eb + 1e-4
    )

    print(f"\nSearch {label}: downhill minima")
    print(f"E(A) = {ea:.6f} eV")
    print(f"E(B) = {eb:.6f} eV")
    print(f"O-site separation = {od:.4f} A")
    print(f"all-atom RMSD = {rmsd:.4f} A")
    print(f"distinct minima? {different_minima}")
    print(f"saddle above both minima? {below_saddle}")

    return {
        **result,
        "minimum_A": a,
        "minimum_B": b,
        "energy_A": ea,
        "energy_B": eb,
        "o_site_distance": od,
        "endpoint_rmsd": rmsd,
        "different_minima": different_minima,
        "saddle_above_both": below_saddle,
    }


# =============================================================================
# 5. NEB BETWEEN THE DIMER-DISCOVERED MINIMA
# =============================================================================

def run_neb(result):
    label = result["label"]
    ndir = OUTDIR / f"neb_path_{label}"
    ndir.mkdir(parents=True, exist_ok=True)

    initial = result["minimum_A"].copy()
    final = result["minimum_B"].copy()

    images = [initial]
    for _ in range(N_NEB_IMAGES - 2):
        images.append(initial.copy())
    images.append(final)

    for image in images:
        attach_emt(image)

    # Smooth IDPP interpolation first.
    neb = NEB(images, k=NEB_SPRING, climb=False)
    neb.interpolate(method="idpp", mic=True)

    opt = FIRE(
        neb,
        trajectory=str(ndir / "neb_relax.traj"),
        logfile=str(ndir / "neb_relax.log"),
    )
    opt.run(
        fmax=max(0.075, 1.5 * NEB_FMAX),
        steps=MAX_NEB_STEPS,
    )

    # Climbing-image refinement.
    neb_ci = NEB(images, k=NEB_SPRING, climb=True)
    opt_ci = FIRE(
        neb_ci,
        trajectory=str(ndir / "neb_climb.traj"),
        logfile=str(ndir / "neb_climb.log"),
    )
    converged = opt_ci.run(
        fmax=NEB_FMAX,
        steps=MAX_NEB_STEPS,
    )

    energies = np.array(
        [img.get_potential_energy() for img in images],
        dtype=float,
    )

    # Reaction coordinate based on accumulated Cartesian path length.
    s = [0.0]
    for i in range(1, len(images)):
        dr = images[i].positions - images[i - 1].positions
        dr_mic, _ = find_mic(dr, images[i].cell, pbc=images[i].pbc)
        ds = float(np.sqrt(np.sum(np.asarray(dr_mic) ** 2)))
        s.append(s[-1] + ds)

    s = np.asarray(s, dtype=float)
    if s[-1] > 0:
        s /= s[-1]
    else:
        s = np.linspace(0.0, 1.0, len(images))

    rel = energies - energies[0]

    # Find the NEB image structurally closest to the Dimer saddle.
    rmsds = np.array(
        [structure_rmsd(img, result["saddle"]) for img in images]
    )
    closest = int(np.argmin(rmsds))

    dimer_x = float(s[closest])
    dimer_y = float(result["saddle_energy"] - energies[0])

    max_idx = int(np.argmax(energies))
    has_interior_maximum = (0 < max_idx < len(images) - 1)

    write(ndir / "neb_images.traj", images)
    np.savetxt(
        ndir / "neb_energies.csv",
        np.column_stack([s, energies, rel]),
        delimiter=",",
        header="reaction_coordinate,energy_eV,relative_to_A_eV",
        comments="",
    )

    print(f"\nNEB search {label}")
    print(f"Converged: {converged}")
    print(f"Interior maximum? {has_interior_maximum}")
    print(f"NEB barrier A->B: {np.max(energies)-energies[0]:.6f} eV")
    print(f"Dimer saddle above A: {dimer_y:.6f} eV")

    return {
        **result,
        "neb_images": images,
        "neb_energies": energies,
        "reaction_coordinate": s,
        "relative_to_A": rel,
        "dimer_plot_x": dimer_x,
        "dimer_plot_y": dimer_y,
        "neb_converged": bool(converged),
        "neb_has_interior_maximum": has_interior_maximum,
        "neb_barrier": float(np.max(energies) - energies[0]),
    }


# =============================================================================
# 6. GROUP NEARLY IDENTICAL NEB PROFILES SO OVERLAPS ARE EXPLICIT
# =============================================================================

def group_neb_profiles(results, tolerance_eV=0.015):
    """
    Group nearly identical energy profiles.

    This prevents a symmetry-equivalent path from silently hiding underneath
    another line.  Each plotted curve is labeled with ALL searches in its group.
    """
    groups = []

    for result in results:
        placed = False
        for group in groups:
            ref = group[0]
            if len(ref["relative_to_A"]) != len(result["relative_to_A"]):
                continue

            diff = np.max(
                np.abs(ref["relative_to_A"] - result["relative_to_A"])
            )
            if diff < tolerance_eV:
                group.append(result)
                placed = True
                break

        if not placed:
            groups.append([result])

    return groups


# =============================================================================
# 7. TWO-PANEL FINAL FIGURE
# =============================================================================

def make_final_figure(
    minimum,
    collective_modes,
    collective_fraction,
    q1,
    q2,
    Z,
    plot_mins,
    plot_maxs,
    projected,
    dimer_results,
    neb_results,
    rejected_transitions,
):
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.4))

    # -------------------------------------------------------------------------
    # LEFT: collective-coordinate PES + only SUCCESSFUL Dimer trajectories
    # -------------------------------------------------------------------------
    ax = axes[0]

    Q1, Q2 = np.meshgrid(q1, q2)
    finite = Z[np.isfinite(Z)]

    # Clip only the extreme repulsive tail so local basin structure is readable.
    display_cap = min(
        float(np.percentile(finite, 90)),
        float(np.nanmin(finite) + 1.25),
    )
    Zshow = np.clip(Z, None, display_cap)

    contour = ax.contourf(
        Q1,
        Q2,
        Zshow,
        levels=26,
        cmap="Greys",
    )
    ax.contour(
        Q1,
        Q2,
        Zshow,
        levels=18,
        colors="0.40",
        linewidths=0.45,
        alpha=0.75,
    )

    cbar = fig.colorbar(
        contour,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Relative energy on collective-coordinate slice (eV)")

    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=110,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
        label="Starting minimum",
    )

    failed_notes = []

    for result in dimer_results:
        label = result["label"]
        xy = projected[label]["trajectory"]
        saddle_xy = projected[label]["saddle"]
        proper_saddle = result["curvature"] < 0

        if proper_saddle:
            # Successful Dimer searches: show the actual local trajectory.
            inside = (
                (xy[:, 0] >= plot_mins[0])
                & (xy[:, 0] <= plot_maxs[0])
                & (xy[:, 1] >= plot_mins[1])
                & (xy[:, 1] <= plot_maxs[1])
            )

            # Keep only the contiguous prefix that stays in the local view.
            # This avoids connecting separated trajectory chunks across a
            # periodic-wrap/runaway excursion.
            first_out = np.where(~inside)[0]
            if len(first_out) > 0:
                stop = max(1, int(first_out[0]))
                plot_xy = xy[:stop]
            else:
                plot_xy = xy

            line, = ax.plot(
                plot_xy[:, 0],
                plot_xy[:, 1],
                marker=".",
                markersize=4,
                linewidth=2.0,
                alpha=0.95,
                label=f"Dimer search {label}",
                zorder=7,
            )

            if (
                plot_mins[0] <= saddle_xy[0] <= plot_maxs[0]
                and plot_mins[1] <= saddle_xy[1] <= plot_maxs[1]
            ):
                ax.scatter(
                    [saddle_xy[0]],
                    [saddle_xy[1]],
                    marker="*",
                    s=210,
                    color=line.get_color(),
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=11,
                )
                ax.annotate(
                    label,
                    saddle_xy,
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        else:
            # Failed searches should not dominate the plot with a runaway path.
            # Show only a gray X at the last local point / saddle projection.
            if (
                plot_mins[0] <= saddle_xy[0] <= plot_maxs[0]
                and plot_mins[1] <= saddle_xy[1] <= plot_maxs[1]
            ):
                ax.scatter(
                    [saddle_xy[0]],
                    [saddle_xy[1]],
                    marker="X",
                    s=105,
                    color="0.45",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=9,
                    label=f"Dimer search {label}: no saddle",
                )
                ax.annotate(
                    label,
                    saddle_xy,
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="0.35",
                )

            failed_notes.append(
                f"{label}: no negative-curvature saddle"
            )

    ax.set_xlim(plot_mins[0], plot_maxs[0])
    ax.set_ylim(plot_mins[1], plot_maxs[1])

    ax.set_xlabel(
        "Collective coordinate 1, $q_1 = \\Delta R \\cdot e_1$ ($\\AA$)"
    )
    ax.set_ylabel(
        "Collective coordinate 2, $q_2 = \\Delta R \\cdot e_2$ ($\\AA$)"
    )
    ax.set_title("(a) Dimer exploration in collective-coordinate space")

    frac1 = collective_fraction[0] * 100.0
    frac2 = collective_fraction[1] * 100.0

    ax.text(
        0.02,
        0.98,
        f"Principal displacement modes: {frac1:.1f}% + {frac2:.1f}% "
        "of sampled Dimer displacement variance",
        transform=ax.transAxes,
        va="top",
        fontsize=7.5,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.82,
            edgecolor="0.7",
        ),
    )

    if failed_notes:
        ax.text(
            0.02,
            0.04,
            "\n".join(failed_notes),
            transform=ax.transAxes,
            va="bottom",
            fontsize=7.3,
            color="0.35",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.82,
                edgecolor="0.75",
            ),
        )

    ax.legend(fontsize=7, loc="best")

    # -------------------------------------------------------------------------
    # RIGHT: plot EVERY valid NEB result separately
    # -------------------------------------------------------------------------
    ax = axes[1]

    if neb_results:
        linestyles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "^", "D", "v", "P", "X"]

        for i, result in enumerate(neb_results):
            label = result["label"]

            line, = ax.plot(
                result["reaction_coordinate"],
                result["relative_to_A"],
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.0,
                markersize=5,
                alpha=0.90,
                label=f"NEB from Dimer search {label}",
            )

            ax.scatter(
                [result["dimer_plot_x"]],
                [result["dimer_plot_y"]],
                marker="*",
                s=210,
                color=line.get_color(),
                edgecolor="black",
                linewidth=0.7,
                zorder=8,
            )

            ax.annotate(
                f"Dimer saddle {label}",
                (
                    result["dimer_plot_x"],
                    result["dimer_plot_y"],
                ),
                xytext=(7, 7 + 10 * (i % 2)),
                textcoords="offset points",
                fontsize=7.5,
                color=line.get_color(),
            )

        # Explicitly call out when profiles overlap numerically.
        overlap_messages = []
        for i in range(len(neb_results)):
            for j in range(i + 1, len(neb_results)):
                a = neb_results[i]
                b = neb_results[j]

                if len(a["relative_to_A"]) != len(b["relative_to_A"]):
                    continue

                diff = float(
                    np.max(
                        np.abs(
                            a["relative_to_A"]
                            - b["relative_to_A"]
                        )
                    )
                )

                if diff < 0.015:
                    overlap_messages.append(
                        f"{a['label']} and {b['label']}: "
                        "symmetry-equivalent / overlapping NEB profiles"
                    )

        if overlap_messages:
            ax.text(
                0.02,
                0.04,
                "\n".join(overlap_messages),
                transform=ax.transAxes,
                va="bottom",
                fontsize=7.3,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.82,
                    edgecolor="0.75",
                ),
            )

    else:
        ax.text(
            0.5,
            0.5,
            "No valid Dimer-discovered transitions\npassed the NEB checks.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Explain every Dimer search that did not generate a NEB curve.
    if rejected_transitions:
        rejected_text = "No NEB generated:\n" + "\n".join(
            f"{item['label']}: {item['reason']}"
            for item in rejected_transitions
        )

        ax.text(
            0.98,
            0.98,
            rejected_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.1,
            color="0.35",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                alpha=0.88,
                edgecolor="0.75",
            ),
        )

    ax.axhline(0.0, linewidth=0.8, color="0.5")
    ax.set_xlabel("Normalized reaction coordinate")
    ax.set_ylabel("Energy relative to initial minimum A (eV)")
    ax.set_title("(b) NEB pathways for Dimer-discovered transitions")
    ax.grid(alpha=0.25)

    if neb_results:
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "O interstitial in FCC Al: Dimer saddle discovery and NEB validation",
        fontsize=15,
    )

    fig.tight_layout()

    path = OUTDIR / "dimer_and_neb_landscape.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)

    print(f"\nSaved final two-panel figure:\n  {path}")


# =============================================================================
# 8. SUMMARY TABLE
# =============================================================================

def write_summary(dimer_results, neb_results):
    neb_by_label = {r["label"]: r for r in neb_results}

    path = OUTDIR / "transition_summary.csv"
    fields = [
        "search",
        "dimer_converged",
        "dimer_curvature",
        "dimer_barrier_from_start_eV",
        "generated_valid_neb",
        "minimum_A_energy_eV",
        "minimum_B_energy_eV",
        "O_site_separation_A",
        "endpoint_RMSD_A",
        "NEB_barrier_A_to_B_eV",
        "Dimer_saddle_above_A_eV",
        "NEB_converged",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for d in dimer_results:
            n = neb_by_label.get(d["label"])

            row = {
                "search": d["label"],
                "dimer_converged": d["converged"],
                "dimer_curvature": d["curvature"],
                "dimer_barrier_from_start_eV": d["barrier_from_initial"],
                "generated_valid_neb": n is not None,
                "minimum_A_energy_eV": "",
                "minimum_B_energy_eV": "",
                "O_site_separation_A": "",
                "endpoint_RMSD_A": "",
                "NEB_barrier_A_to_B_eV": "",
                "Dimer_saddle_above_A_eV": "",
                "NEB_converged": "",
            }

            if n is not None:
                row.update(
                    {
                        "minimum_A_energy_eV": n["energy_A"],
                        "minimum_B_energy_eV": n["energy_B"],
                        "O_site_separation_A": n["o_site_distance"],
                        "endpoint_RMSD_A": n["endpoint_rmsd"],
                        "NEB_barrier_A_to_B_eV": n["neb_barrier"],
                        "Dimer_saddle_above_A_eV": n["dimer_plot_y"],
                        "NEB_converged": n["neb_converged"],
                    }
                )

            writer.writerow(row)

    print(f"Saved summary table:\n  {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(
        "\n"
        "========================================================================\n"
        " DIMER + NEB DEMO: O INTERSTITIAL IN BULK Al\n"
        "========================================================================\n"
        "EMT/O WARNING: this is an algorithm demo, not a quantitatively reliable\n"
        "model of oxygen diffusion in aluminum.\n"
    )

    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True)

    # Starting minimum
    minimum, minimum_energy, o_index = build_starting_minimum()

    # Run ALL Dimer searches first.  Their actual many-atom trajectories define
    # the collective coordinates used for the left-hand landscape.
    dimer_results = []

    for label, direction in DIMER_DIRECTIONS.items():
        try:
            result = run_dimer(
                minimum,
                minimum_energy,
                o_index,
                label,
                direction,
            )
            dimer_results.append(result)
        except Exception as exc:
            print(f"\nWARNING: Dimer search {label} failed:")
            print(f"{type(exc).__name__}: {exc}")

    # Build a collective-coordinate representation from the actual Dimer data.
    collective_modes, collective_fraction = fit_collective_modes(
        minimum,
        dimer_results,
    )
    projected = project_dimer_results(
        minimum,
        dimer_results,
        collective_modes,
    )
    q1, q2, Z, plot_mins, plot_maxs = build_collective_coordinate_pes(
        minimum,
        minimum_energy,
        collective_modes,
        projected,
        dimer_results,
    )

    # Discover endpoints and run NEB only for physically usable saddles.
    # Keep explicit reasons for every Dimer search that does not become a
    # valid NEB pathway so the final figure can teach from those failures.
    neb_results = []
    rejected_transitions = []

    for result in dimer_results:
        if result["curvature"] >= 0:
            reason = "no negative-curvature saddle found"
            print(
                f"\nSkipping NEB for {result['label']}: {reason}."
            )
            rejected_transitions.append(
                {"label": result["label"], "reason": reason}
            )
            continue

        try:
            connected = relax_both_sides(result, o_index)

            if not connected["different_minima"]:
                reason = "both downhill directions relaxed to the same minimum"
                print(
                    f"\nSkipping NEB for {result['label']}: {reason}."
                )
                rejected_transitions.append(
                    {"label": result["label"], "reason": reason}
                )
                continue

            if not connected["saddle_above_both"]:
                reason = "Dimer saddle was not above both relaxed minima"
                print(
                    f"\nSkipping NEB for {result['label']}: {reason}."
                )
                rejected_transitions.append(
                    {"label": result["label"], "reason": reason}
                )
                continue

            neb_result = run_neb(connected)

            if not neb_result["neb_has_interior_maximum"]:
                reason = "NEB did not contain an interior energy maximum"
                print(
                    f"\nSkipping final NEB display for "
                    f"{result['label']}: {reason}."
                )
                rejected_transitions.append(
                    {"label": result["label"], "reason": reason}
                )
                continue

            neb_results.append(neb_result)

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"\nWARNING: transition {result['label']} failed:")
            print(reason)
            rejected_transitions.append(
                {"label": result["label"], "reason": reason}
            )

    make_final_figure(
        minimum,
        collective_modes,
        collective_fraction,
        q1,
        q2,
        Z,
        plot_mins,
        plot_maxs,
        projected,
        dimer_results,
        neb_results,
        rejected_transitions,
    )

    write_summary(dimer_results, neb_results)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

    print(f"\nDimer searches attempted: {len(DIMER_DIRECTIONS)}")
    print(f"Dimer searches completed: {len(dimer_results)}")
    print(f"Valid NEB transitions:    {len(neb_results)}")

    for r in neb_results:
        print(
            f"  {r['label']:>4s}: "
            f"Dimer = {r['dimer_plot_y']:.4f} eV above A, "
            f"NEB = {r['neb_barrier']:.4f} eV"
        )

    print(
        "\nMain figure:\n"
        f"  {OUTDIR / 'dimer_and_neb_landscape.png'}\n"
    )


if __name__ == "__main__":
    main()
