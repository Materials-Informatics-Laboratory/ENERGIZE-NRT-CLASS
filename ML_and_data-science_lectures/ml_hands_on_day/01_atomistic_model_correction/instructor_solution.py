#!/usr/bin/env python3
"""
ME500 Hands-On Day 1
Problem 1 — Instructor Reference Solution
Correcting an Imperfect Atomistic Model

This version exposes a few DATA-DIFFICULTY settings near the top of the file.
You can make the raw simulation-to-experiment comparison cleaner or noisier
without changing the ML workflow itself.

Reference workflow:
    1. Generate/load synthetic simulation + experimental data.
    2. Establish the raw atomistic prediction as the baseline.
    3. Learn the simulation-to-experiment discrepancy from atomistic descriptors.
    4. Evaluate only on a held-out test set.
    5. Compare baseline vs ML-corrected predictions using metrics and figures.
    6. Save the full preprocessing/model pipeline and test predictions.

This is ONE defensible solution, not the only correct student solution.
"""

from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 0. ROBUST FILE MANAGEMENT
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURE_DIR = SCRIPT_DIR / "figures"
MODEL_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

for directory in (DATA_DIR, FIGURE_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "atomistic_vs_experiment.csv"
METADATA_FILE = DATA_DIR / "dataset_metadata.json"

# -----------------------------------------------------------------------------
# 1. DATA-DIFFICULTY SETTINGS
# -----------------------------------------------------------------------------
# These are intentionally collected in one place so the instructor can tune the
# exercise without editing the generator or ML workflow.
#
# SYSTEMATIC_DISCREPANCY_STRENGTH controls the magnitude of the *learnable*
# simulation-to-experiment bias. Larger values make the raw atomistic model
# worse while still leaving a pattern that ML can learn from the descriptors.
#
# RANDOM_SIM_TO_EXP_NOISE_GPA controls the *unlearnable* random scatter between
# simulation and experiment. Larger values make the raw parity plot noisier and
# also place a realistic floor on how accurate the corrected model can become.
#
# Recommended classroom default:
#     SYSTEMATIC_DISCREPANCY_STRENGTH = 1.50
#     RANDOM_SIM_TO_EXP_NOISE_GPA = 6.0
# This typically produces a visibly imperfect raw model while retaining a clear
# ML improvement.
RANDOM_SEED = 500
N_MATERIALS = 320
SYSTEMATIC_DISCREPANCY_STRENGTH = 1.50
RANDOM_SIM_TO_EXP_NOISE_GPA = 10.0

# Set True to force regeneration every time. Normally False is convenient.
# Even when False, the script automatically regenerates the dataset if any of
# the data-generation settings above have changed since the CSV was created.
REGENERATE_DATA = False
AUTO_REGENERATE_IF_SETTINGS_CHANGE = True


def save_figure(fig, filename: str) -> Path:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def save_model(model, filename="discrepancy_model.joblib") -> Path:
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    print(f"Saved model: {path}")
    return path


def load_model(filename="discrepancy_model.joblib"):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No saved model was found at {path}")
    return joblib.load(path)


def generation_settings() -> dict:
    """Return only the settings that determine the generated dataset."""
    return {
        "seed": int(RANDOM_SEED),
        "n_materials": int(N_MATERIALS),
        "systematic_discrepancy_strength": float(SYSTEMATIC_DISCREPANCY_STRENGTH),
        "random_sim_to_exp_noise_GPa": float(RANDOM_SIM_TO_EXP_NOISE_GPA),
    }


def saved_settings_match_current() -> bool:
    """Check whether the existing CSV was generated with the current settings."""
    if not DATA_FILE.exists() or not METADATA_FILE.exists():
        return False

    try:
        metadata = json.loads(METADATA_FILE.read_text())
        return metadata.get("generation_settings") == generation_settings()
    except (OSError, json.JSONDecodeError):
        return False


# -----------------------------------------------------------------------------
# 2. SYNTHETIC DATA GENERATOR
# -----------------------------------------------------------------------------
def generate_atomistic_dataset(path: Path) -> None:
    """Create deterministic atomistic + experimental classroom data."""
    rng = np.random.default_rng(RANDOM_SEED)
    n_materials = N_MATERIALS

    # Latent variables exist only to create physically sensible correlations.
    bonding_strength = rng.uniform(0.15, 1.0, n_materials)
    packing = rng.uniform(0.20, 1.0, n_materials)
    directionality = rng.uniform(0.0, 1.0, n_materials)

    cohesive_energy = 1.2 + 5.8 * bonding_strength + rng.normal(0, 0.18, n_materials)
    atomic_volume = 25.0 - 9.0 * packing - 2.0 * bonding_strength + rng.normal(0, 0.7, n_materials)
    atomic_volume = np.clip(atomic_volume, 8.5, 25.0)

    coordination_number = np.clip(
        4.0 + 8.0 * packing + rng.normal(0, 0.45, n_materials),
        3.0,
        12.5,
    )

    mean_bond_length = (
        3.05
        - 0.48 * packing
        - 0.22 * bonding_strength
        + rng.normal(0, 0.035, n_materials)
    )

    bulk_modulus = (
        18.0
        + 34.0 * cohesive_energy
        + 22.0 * packing
        - 2.0 * atomic_volume
        + rng.normal(0, 8.0, n_materials)
    )
    bulk_modulus = np.clip(bulk_modulus, 15.0, 300.0)

    shear_modulus = (
        0.47 * bulk_modulus
        + 22.0 * directionality
        - 7.0
        + rng.normal(0, 6.0, n_materials)
    )
    shear_modulus = np.clip(shear_modulus, 7.0, 190.0)

    # Imperfect atomistic prediction of Young's modulus.
    young_sim = 9.0 * bulk_modulus * shear_modulus / (3.0 * bulk_modulus + shear_modulus)
    young_sim += rng.normal(0, 5.0, n_materials)

    # A descriptor-dependent discrepancy that ML can, in principle, learn.
    base_systematic_correction = (
        34.0
        - 0.16 * young_sim
        + 10.0 * (cohesive_energy - cohesive_energy.mean())
        - 3.2 * (atomic_volume - atomic_volume.mean())
        + 0.11 * (bulk_modulus - bulk_modulus.mean())
        + 2.8 * (coordination_number - coordination_number.mean())
        + 9.0 * np.sin(1.7 * mean_bond_length)
    )

    systematic_correction = (
        SYSTEMATIC_DISCREPANCY_STRENGTH * base_systematic_correction
    )

    # Random simulation-to-experiment mismatch. This term is intentionally not
    # predictable from the provided descriptors.
    random_discrepancy = rng.normal(
        0.0,
        RANDOM_SIM_TO_EXP_NOISE_GPA,
        n_materials,
    )

    young_exp = np.clip(
        young_sim + systematic_correction + random_discrepancy,
        8.0,
        None,
    )

    df = pd.DataFrame({
        "material_id": [f"MAT_{i:04d}" for i in range(n_materials)],
        "cohesive_energy_eV_per_atom": cohesive_energy,
        "atomic_volume_A3_per_atom": atomic_volume,
        "coordination_number": coordination_number,
        "mean_bond_length_A": mean_bond_length,
        "bulk_modulus_sim_GPa": bulk_modulus,
        "shear_modulus_sim_GPa": shear_modulus,
        "youngs_modulus_sim_GPa": young_sim,
        "youngs_modulus_exp_GPa": young_exp,
    })
    df.to_csv(path, index=False)

    metadata = {
        "description": (
            "Synthetic atomistic descriptors plus simulated and experimental "
            "Young's modulus for the ME500 ML hands-on exercise."
        ),
        "generation_settings": generation_settings(),
        "units": {
            "cohesive_energy_eV_per_atom": "eV/atom",
            "atomic_volume_A3_per_atom": "angstrom^3/atom",
            "coordination_number": "dimensionless",
            "mean_bond_length_A": "angstrom",
            "bulk_modulus_sim_GPa": "GPa",
            "shear_modulus_sim_GPa": "GPa",
            "youngs_modulus_sim_GPa": "GPa",
            "youngs_modulus_exp_GPa": "GPa",
        },
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    print(f"Generated dataset: {path}")
    print(
        "Data settings: "
        f"systematic strength={SYSTEMATIC_DISCREPANCY_STRENGTH:.2f}, "
        f"random noise={RANDOM_SIM_TO_EXP_NOISE_GPA:.1f} GPa"
    )


def load_dataset() -> pd.DataFrame:
    should_regenerate = REGENERATE_DATA or not DATA_FILE.exists()

    if AUTO_REGENERATE_IF_SETTINGS_CHANGE and not saved_settings_match_current():
        should_regenerate = True

    if should_regenerate:
        generate_atomistic_dataset(DATA_FILE)
    else:
        print(f"Reusing existing dataset: {DATA_FILE}")

    return pd.read_csv(DATA_FILE)


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE_GPa": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_GPa": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# -----------------------------------------------------------------------------
# 3. COMPLETE REFERENCE WORKFLOW
# -----------------------------------------------------------------------------
def main() -> None:
    df = load_dataset()
    print(f"Loaded {len(df)} materials from {DATA_FILE}")

    # The final quantity of interest is the experimental Young's modulus.
    # Here we use discrepancy learning:
    #
    #   experimental = atomistic prediction + learned correction
    #
    # Therefore the regression target is the atomistic model's error.
    df["simulation_error_GPa"] = (
        df["youngs_modulus_exp_GPa"] - df["youngs_modulus_sim_GPa"]
    )

    # All selected features are quantities that would be available from the
    # atomistic calculation before consulting the experimental answer.
    feature_columns = [
        "cohesive_energy_eV_per_atom",
        "atomic_volume_A3_per_atom",
        "coordination_number",
        "mean_bond_length_A",
        "bulk_modulus_sim_GPa",
        "shear_modulus_sim_GPa",
        "youngs_modulus_sim_GPa",
    ]

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=RANDOM_SEED,
    )

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train_error = train_df["simulation_error_GPa"]

    y_test_exp = test_df["youngs_modulus_exp_GPa"].to_numpy()
    y_test_sim = test_df["youngs_modulus_sim_GPa"].to_numpy()

    # BASELINE: simply trust the raw atomistic prediction.
    baseline_metrics = metrics(y_test_exp, y_test_sim)

    # The entire preprocessing + model workflow is stored in one Pipeline.
    # Scaling is not needed by RandomForestRegressor, but keeping preprocessing
    # inside the saved pipeline makes the workflow reproducible and easy to swap
    # for a scale-sensitive model later.
    preprocess = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), feature_columns)],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("regressor", model),
    ])

    pipeline.fit(X_train, y_train_error)

    predicted_error = pipeline.predict(X_test)
    y_test_corrected = y_test_sim + predicted_error
    corrected_metrics = metrics(y_test_exp, y_test_corrected)

    print("\nHeld-out test performance")
    print("-------------------------")
    print("Raw atomistic baseline:")
    print(json.dumps(baseline_metrics, indent=2))
    print("ML-corrected prediction:")
    print(json.dumps(corrected_metrics, indent=2))

    # Save and immediately reload the model to verify that the serialized
    # preprocessing + regression workflow reproduces the same predictions.
    save_model(pipeline)
    reloaded = load_model()
    assert np.allclose(reloaded.predict(X_test), predicted_error)

    pred_table = pd.DataFrame({
        "material_id": test_df["material_id"].to_numpy(),
        "experimental_GPa": y_test_exp,
        "atomistic_GPa": y_test_sim,
        "learned_correction_GPa": predicted_error,
        "corrected_GPa": y_test_corrected,
        "baseline_abs_error_GPa": np.abs(y_test_sim - y_test_exp),
        "corrected_abs_error_GPa": np.abs(y_test_corrected - y_test_exp),
    })
    pred_table.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    summary = {
        "generation_settings": generation_settings(),
        "feature_columns": feature_columns,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "baseline": baseline_metrics,
        "ml_corrected": corrected_metrics,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: direct side-by-side baseline vs corrected comparison.
    low = min(y_test_exp.min(), y_test_sim.min(), y_test_corrected.min()) - 5
    high = max(y_test_exp.max(), y_test_sim.max(), y_test_corrected.max()) + 5

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(y_test_exp, y_test_sim, alpha=0.75)
    axes[0].plot([low, high], [low, high], "--", linewidth=1.5)
    axes[0].set_xlim(low, high)
    axes[0].set_ylim(low, high)
    axes[0].set_xlabel("Experimental Young's modulus (GPa)")
    axes[0].set_ylabel("Atomistic prediction (GPa)")
    axes[0].set_title(
        f"Raw simulation\nMAE = {baseline_metrics['MAE_GPa']:.1f} GPa"
    )

    axes[1].scatter(y_test_exp, y_test_corrected, alpha=0.75)
    axes[1].plot([low, high], [low, high], "--", linewidth=1.5)
    axes[1].set_xlim(low, high)
    axes[1].set_ylim(low, high)
    axes[1].set_xlabel("Experimental Young's modulus (GPa)")
    axes[1].set_ylabel("ML-corrected prediction (GPa)")
    axes[1].set_title(
        "Simulation + learned correction\n"
        f"MAE = {corrected_metrics['MAE_GPa']:.1f} GPa"
    )

    fig.suptitle("Can ML improve an imperfect atomistic model?")
    fig.tight_layout()
    save_figure(fig, "baseline_vs_corrected_parity.png")

    # Figure 2: residual error after correction.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test_sim, y_test_corrected - y_test_exp, alpha=0.75)
    ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Atomistic Young's modulus (GPa)")
    ax.set_ylabel("Corrected prediction error (GPa)")
    ax.set_title("Residual error after ML correction")
    fig.tight_layout()
    save_figure(fig, "corrected_residuals.png")

    worst = pred_table.sort_values(
        "corrected_abs_error_GPa",
        ascending=False,
    ).head(8)

    print("\nLargest remaining corrected errors:")
    print(
        worst[
            [
                "material_id",
                "experimental_GPa",
                "corrected_GPa",
                "corrected_abs_error_GPa",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
