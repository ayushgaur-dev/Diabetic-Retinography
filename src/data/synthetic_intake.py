"""SYNTHETIC DATA — DEMONSTRATION ONLY. NOT USED FOR TRAINING OR EVALUATION.

Generates synthetic patient intake records (age, diabetes duration,
HbA1c, visual acuity, spherical refraction) to pre-fill the Streamlit
demo's intake form (Block 18) so the app runs standalone without real
patient data. These rows never touch the vision model's training,
validation, or test pipeline, and never feed the rule engine or RAG
evaluation sets used elsewhere in this project — they exist solely to
demonstrate the UI.

Distributions are chosen for plausibility (right-skewed diabetes
duration, bimodal refraction, etc.), not fit to any real population —
do not use this to draw clinical or epidemiological conclusions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "synthetic"
OUTPUT_FILENAME = "synthetic_intake_demo.csv"  # filename itself signals synthetic origin

N_ROWS = 100
SEED = 42


def _truncated_normal(rng, mean, std, low, high, size):
    """Draw `size` samples from N(mean, std), resampling any value that
    falls outside [low, high] until all satisfy the bound.

    Preferred over np.clip() here: clipping piles up probability mass
    exactly at the boundary, which a real bounded clinical measurement
    wouldn't do; resampling keeps the shape of the (truncated) normal
    instead.
    """
    samples = rng.normal(mean, std, size)
    out_of_bounds = (samples < low) | (samples > high)
    while np.any(out_of_bounds):
        samples[out_of_bounds] = rng.normal(mean, std, int(out_of_bounds.sum()))
        out_of_bounds = (samples < low) | (samples > high)
    return samples


def generate_synthetic_intake(n_rows=N_ROWS, seed=SEED):
    """Generate n_rows of synthetic patient intake data as a DataFrame.

    Columns:
        patient_id            str, e.g. "SYNTH-0001" — prefix makes the
                               synthetic origin visible even if the
                               column headers are stripped downstream.
        age                    float, years, N(60, 12) bounded [18, 90].
        diabetes_years         int, years since diagnosis. Right-skewed
                               (gamma-distributed) before being capped so
                               that age - diabetes_years >= 12, i.e. no
                               implied diabetes onset before age 12.
        hba1c                  float, %, N(7.8, 1.4) bounded [5.0, 14.0].
        visual_acuity           float, decimal notation (1.0 = normal),
                               left-skewed toward the upper end of
                               [0.05, 1.0] — most patients cluster near
                               good acuity with a tail toward poor acuity.
        spherical_refraction    float, dioptres, bimodal: ~85% a tight
                               near-emmetropic cluster around 0 D, ~15% a
                               highly myopic cluster around -8 D, overall
                               bounded [-12.0, +6.0].
    """
    rng = np.random.default_rng(seed)

    age = _truncated_normal(rng, mean=60, std=12, low=18, high=90, size=n_rows)
    age = np.round(age, 1)  # rounded now so the cap below matches the final age column exactly

    # Diabetes duration: right-skewed (many recently diagnosed, fewer
    # decades-long cases), then capped per-row so implied diagnosis age
    # never drops below 12. Uses floor (not round) after capping —
    # rounding to nearest can round UP past the cap (e.g. a capped value
    # of 8.6 would round to 9, breaching the >=12 constraint by 0.4).
    years_diabetes_raw = rng.gamma(shape=2.0, scale=5.0, size=n_rows)
    max_years_diabetes = np.maximum(age - 12.0, 0.0)
    diabetes_years = np.minimum(years_diabetes_raw, max_years_diabetes)
    diabetes_years = np.floor(diabetes_years).astype(int)

    hba1c = _truncated_normal(rng, mean=7.8, std=1.4, low=5.0, high=14.0, size=n_rows)

    # Visual acuity: Beta(5, 2) is left-skewed (mode near its upper end),
    # scaled onto [0.05, 1.0] so most patients land near good acuity
    # (close to 1.0) with a tail stretching down toward poor acuity.
    va_low, va_high = 0.05, 1.0
    visual_acuity = va_low + (va_high - va_low) * rng.beta(a=5.0, b=2.0, size=n_rows)

    # Spherical refraction: bimodal mixture — most patients drawn from a
    # tight near-emmetropic cluster, a smaller subset from a separate
    # highly-myopic cluster, both truncated to the same overall bound.
    is_myopic_cluster = rng.random(n_rows) < 0.15
    refraction = np.empty(n_rows)
    n_myopic = int(is_myopic_cluster.sum())
    n_emmetropic = n_rows - n_myopic
    refraction[~is_myopic_cluster] = _truncated_normal(
        rng, mean=0.0, std=1.75, low=-12.0, high=6.0, size=n_emmetropic
    )
    refraction[is_myopic_cluster] = _truncated_normal(
        rng, mean=-8.0, std=2.5, low=-12.0, high=6.0, size=n_myopic
    )

    patient_id = [f"SYNTH-{i + 1:04d}" for i in range(n_rows)]

    return pd.DataFrame({
        "patient_id": patient_id,
        "age": age,
        "diabetes_years": diabetes_years,
        "hba1c": np.round(hba1c, 1),
        "visual_acuity": np.round(visual_acuity, 2),
        "spherical_refraction": np.round(refraction, 2),
    })


if __name__ == "__main__":
    df = generate_synthetic_intake()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    df.to_csv(output_path, index=False)

    print("SYNTHETIC DATA — DEMONSTRATION ONLY. NOT USED FOR TRAINING OR EVALUATION.")
    print(f"Saved {len(df)} rows to: {output_path}\n")

    print("Summary statistics (sanity check on distributions):")
    print(df.describe())

    # Spot-check the age/diabetes_years constraint held for every row.
    diagnosis_age = df["age"] - df["diabetes_years"]
    print(f"\nMin implied diagnosis age: {diagnosis_age.min():.1f} (must be >= 12)")

    print("\nMyopic-cluster share (spherical_refraction <= -6.0):",
          f"{(df['spherical_refraction'] <= -6.0).mean():.1%}")
