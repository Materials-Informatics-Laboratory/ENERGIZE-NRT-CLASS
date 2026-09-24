#!/usr/bin/env python3
"""
ME500 Hands-On Day 1
Problem 2 — Instructor Reference Solution
Which Simulated Atomic Structure Matches Experiment?

Reference workflow:
    1. Generate/load labeled simulated RDFs and unlabeled experimental-like RDFs.
    2. Split only the simulated data into train/test sets.
    3. Standardize RDF features, reduce dimension with PCA, classify with k-NN.
    4. Evaluate on held-out simulated structures.
    5. Refit/use the same pipeline for experimental-like samples.
    6. Visualize simulation + experiment in a common PCA space.

This is one defensible solution, not the only correct solution.
"""

from pathlib import Path
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FIGURE_DIR = SCRIPT_DIR / "figures"
MODEL_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
for directory in (DATA_DIR, FIGURE_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SIM_FILE = DATA_DIR / "simulated_rdfs.csv"
EXP_FILE = DATA_DIR / "experimental_rdfs.csv"
GRID_FILE = DATA_DIR / "rdf_grid.csv"
RANDOM_SEED = 801
REGENERATE_DATA = False


def save_figure(fig, filename):
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
    return path


def _gaussian(r, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((r - center) / width) ** 2)


def _base_rdf(r, state_index, rng, experimental=False):
    shift_scale = 0.025 if not experimental else 0.075
    width_scale = rng.uniform(0.92, 1.12) if not experimental else rng.uniform(1.15, 1.48)
    noise_scale = 0.045 if not experimental else 0.085
    global_shift = rng.normal(0, shift_scale)
    g = np.ones_like(r)

    if state_index == 0:
        peaks = [(2.45, 0.11, 2.7), (3.48, 0.15, 1.7), (4.26, 0.18, 1.4), (4.92, 0.21, 1.0), (5.55, 0.24, 0.72), (6.15, 0.28, 0.52)]
    elif state_index == 1:
        peaks = [(2.47, 0.16, 2.35), (3.50, 0.21, 1.38), (4.30, 0.25, 1.08), (4.98, 0.29, 0.75), (5.62, 0.33, 0.46)]
    elif state_index == 2:
        peaks = [(2.52, 0.23, 1.85), (4.18, 0.43, 0.72), (5.80, 0.62, 0.32)]
    else:
        peaks = [(2.60, 0.31, 1.45), (4.65, 0.62, 0.43)]

    for center, width, amp in peaks:
        local_shift = global_shift + rng.normal(0, shift_scale * 0.35)
        local_width = width * width_scale * rng.uniform(0.94, 1.08)
        local_amp = amp * rng.uniform(0.90, 1.10)
        g += _gaussian(r, center + local_shift, local_width, local_amp)

    g *= 1.0 / (1.0 + np.exp(-(r - 2.05) / 0.08))
    if state_index >= 2:
        g += 0.11 * np.exp(-0.55 * (r - 2.2)) * np.sin(5.2 * (r - 2.2))
    if experimental:
        g = rng.uniform(0.90, 1.10) * g + rng.normal(0, 0.025) * (r - r.mean())
    g += rng.normal(0, noise_scale, len(r))
    return np.clip(g, 0.0, None)


def generate_rdf_datasets(seed=RANDOM_SEED, n_per_state=70, n_experimental=12):
    rng = np.random.default_rng(seed)
    r = np.linspace(1.6, 7.8, 110)
    state_names = np.array(["crystal", "defect_rich_crystal", "amorphous", "liquid"])
    feature_names = [f"g_r_{value:.3f}" for value in r]

    sim_rows = []
    counter = 0
    for state_index, state_name in enumerate(state_names):
        for _ in range(n_per_state):
            curve = _base_rdf(r, state_index, rng, experimental=False)
            row = {"sample_id": f"SIM_{counter:04d}", "state": state_name}
            row.update(dict(zip(feature_names, curve)))
            sim_rows.append(row)
            counter += 1

    experimental_state_indices = rng.integers(0, len(state_names), size=n_experimental)
    exp_rows = []
    truth_rows = []
    for i, state_index in enumerate(experimental_state_indices):
        curve = _base_rdf(r, int(state_index), rng, experimental=True)
        sample_id = f"EXP_{i:03d}"
        row = {"sample_id": sample_id}
        row.update(dict(zip(feature_names, curve)))
        exp_rows.append(row)
        truth_rows.append({"sample_id": sample_id, "true_state": state_names[int(state_index)]})

    pd.DataFrame(sim_rows).to_csv(SIM_FILE, index=False)
    pd.DataFrame(exp_rows).to_csv(EXP_FILE, index=False)
    pd.DataFrame({"r_A": r, "rdf_column": feature_names}).to_csv(GRID_FILE, index=False)
    # Truth is written only to the instructor results folder, not to the student data.
    pd.DataFrame(truth_rows).to_csv(RESULTS_DIR / "instructor_experimental_truth.csv", index=False)

    metadata = {
        "seed": seed,
        "simulated_samples_per_state": n_per_state,
        "experimental_samples": n_experimental,
        "candidate_states": state_names.tolist(),
    }
    (DATA_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))


def load_datasets():
    truth_file = RESULTS_DIR / "instructor_experimental_truth.csv"
    if REGENERATE_DATA or not (SIM_FILE.exists() and EXP_FILE.exists() and GRID_FILE.exists() and truth_file.exists()):
        generate_rdf_datasets()
    sim_df = pd.read_csv(SIM_FILE)
    exp_df = pd.read_csv(EXP_FILE)
    r = pd.read_csv(GRID_FILE)["r_A"].to_numpy(dtype=float)
    truth_df = pd.read_csv(truth_file)
    return sim_df, exp_df, r, truth_df


def main():
    sim_df, exp_df, r, truth_df = load_datasets()
    feature_columns = [c for c in sim_df.columns if c.startswith("g_r_")]

    X = sim_df[feature_columns].to_numpy(dtype=float)
    y = sim_df["state"].to_numpy()
    X_exp = exp_df[feature_columns].to_numpy(dtype=float)

    # Stratification keeps every structural state represented in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    # One compact Pipeline guarantees that exactly the same scaling/PCA mapping is
    # applied during training, simulated testing, and experimental prediction.
    classifier = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.97, random_state=RANDOM_SEED)),
        ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ])
    classifier.fit(X_train, y_train)

    y_test_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Held-out simulated accuracy: {test_accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred))

    joblib.dump(classifier, MODEL_DIR / "structure_classifier.joblib")

    # Predict the experimental-like measurements only AFTER checking simulation
    # generalization. The experimental labels are not used for fitting.
    exp_pred = classifier.predict(X_exp)
    pred_df = pd.DataFrame({
        "sample_id": exp_df["sample_id"],
        "predicted_state": exp_pred,
    })
    pred_df.to_csv(RESULTS_DIR / "experimental_predictions.csv", index=False)

    checked = pred_df.merge(truth_df, on="sample_id", how="left")
    checked["correct"] = checked["predicted_state"] == checked["true_state"]
    experimental_accuracy = checked["correct"].mean()
    checked.to_csv(RESULTS_DIR / "instructor_checked_experimental_predictions.csv", index=False)
    print(f"Experimental-like hidden-label accuracy: {experimental_accuracy:.3f}")

    metrics = {
        "held_out_simulation_accuracy": float(test_accuracy),
        "experimental_hidden_label_accuracy": float(experimental_accuracy),
        "n_simulated": int(len(sim_df)),
        "n_experimental": int(len(exp_df)),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix for the honest simulated test set.
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=ax, xticks_rotation=25)
    ax.set_title("Held-out simulated structure classification")
    fig.tight_layout()
    save_figure(fig, "simulation_confusion_matrix.png")

    # Build a 2D PCA visualization using the scaler learned on the training data.
    scaler = classifier.named_steps["scale"]
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_exp_scaled = scaler.transform(X_exp)

    pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
    train_2d = pca2.fit_transform(X_train_scaled)
    test_2d = pca2.transform(X_test_scaled)
    exp_2d = pca2.transform(X_exp_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    states = np.unique(y_train)
    for state in states:
        mask = y_train == state
        ax.scatter(train_2d[mask, 0], train_2d[mask, 1], alpha=0.45, label=f"sim: {state}")
    ax.scatter(exp_2d[:, 0], exp_2d[:, 1], marker="*", s=180, edgecolors="black", linewidths=0.8, label="experimental")
    for i, sample_id in enumerate(exp_df["sample_id"]):
        ax.annotate(sample_id, (exp_2d[i, 0], exp_2d[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Simulated and experimental RDFs in a common representation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "simulation_experiment_pca.png")

    # Representative RDF comparison for the experimental samples.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i in range(min(8, len(exp_df))):
        ax.plot(r, X_exp[i], alpha=0.75, label=f"{exp_df.iloc[i]['sample_id']} -> {exp_pred[i]}")
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("Experimental-like RDFs and predicted states")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, "experimental_rdf_predictions.png")

    print("\nExperimental predictions:")
    print(checked.to_string(index=False))


if __name__ == "__main__":
    main()
