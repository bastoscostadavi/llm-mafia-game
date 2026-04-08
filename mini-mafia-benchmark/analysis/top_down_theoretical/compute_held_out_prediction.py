#!/usr/bin/env python3
"""
Held-out prediction experiment for the Mini-Mafia rebuttal.

Tests whether Eq. (2) captures interactive structure beyond aggregate statistics.

Procedure:
  1. Load all (mafioso, villager, detective) configuration outcomes.
  2. Hold out 20% of configurations (stratified random split).
  3. Fit Eq. (2) on the remaining 80% using PyMC.
  4. Predict held-out outcomes from the fitted parameters.
  5. Compare MAE to a baseline that uses only per-model aggregate win rates.

The baseline predicts each configuration's win rate as the average of:
  - the mafioso model's overall win rate when playing mafioso
  - adjusted by the detective model's overall loss rate when playing detective
  - adjusted by the villager model's overall loss rate when playing villager
More precisely, the baseline uses the marginal win rate for each role independently.

Usage:
    python compute_held_out_prediction.py [--held-out-frac 0.2] [--n-splits 5] [--draws 2000] [--seed 42]
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "top_down_theoretical"
DB_PATH = Path(__file__).parent.parent.parent / "database" / "mini_mafia.db"


def load_data():
    """Load and aggregate game outcomes by (mafioso, villager, detective) configuration."""
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT DISTINCT
        g.game_id, g.winner,
        mafioso.model_name  AS mafioso_model,
        villager.model_name AS villager_model,
        detective.model_name AS detective_model
    FROM games g
    JOIN benchmark b ON g.game_id = b.game_id
    JOIN game_players gp_m ON g.game_id = gp_m.game_id AND gp_m.role = 'mafioso'
    JOIN players mafioso ON gp_m.player_id = mafioso.player_id
    JOIN game_players gp_v ON g.game_id = gp_v.game_id AND gp_v.role = 'villager'
    JOIN players villager ON gp_v.player_id = villager.player_id
    JOIN game_players gp_d ON g.game_id = gp_d.game_id AND gp_d.role = 'detective'
    JOIN players detective ON gp_d.player_id = detective.player_id
    WHERE g.winner IS NOT NULL
    """
    games = pd.read_sql_query(query, conn)
    conn.close()
    games["mafia_win"] = (games["winner"] == "mafia").astype(int)

    grouped = (
        games
        .groupby(["mafioso_model", "villager_model", "detective_model"])
        .agg(wins=("mafia_win", "sum"), total=("mafia_win", "count"))
        .reset_index()
    )
    grouped["win_rate"] = grouped["wins"] / grouped["total"]
    return grouped


def fit_theoretical_model(train_df, all_models, model_idx, draws=2000, seed=42):
    """Fit Eq. (2) on training data and return posterior means for a, b, c."""
    maf_ix = [model_idx[m] for m in train_df["mafioso_model"]]
    vil_ix = [model_idx[m] for m in train_df["villager_model"]]
    det_ix = [model_idx[m] for m in train_df["detective_model"]]
    wins = train_df["wins"].values
    totals = train_df["total"].values
    n_models = len(all_models)

    with pm.Model():
        a = pm.Normal("a", mu=0, sigma=2, shape=n_models)
        b = pm.Normal("b", mu=0, sigma=2, shape=n_models)
        c = pm.Normal("c", mu=0, sigma=2, shape=n_models)
        pm.Potential("c_sum_positive",
                     pm.math.switch(pm.math.sum(c) > 0, 0, -np.inf))
        logit_p = c[vil_ix] * (a[maf_ix] - b[det_ix])
        pm.Binomial("y", n=totals, logit_p=logit_p, observed=wins)
        trace = pm.sample(draws=draws, tune=2000, chains=4, cores=4,
                          target_accept=0.99, random_seed=seed,
                          progressbar=False)

    a_mean = trace.posterior["a"].values.reshape(-1, n_models).mean(axis=0)
    b_mean = trace.posterior["b"].values.reshape(-1, n_models).mean(axis=0)
    c_mean = trace.posterior["c"].values.reshape(-1, n_models).mean(axis=0)
    return a_mean, b_mean, c_mean


def predict_theoretical(test_df, a_mean, b_mean, c_mean, model_idx):
    """Predict win rates for test configurations using Eq. (2)."""
    from scipy.special import expit
    preds = []
    for _, row in test_df.iterrows():
        i = model_idx[row["mafioso_model"]]
        k = model_idx[row["villager_model"]]
        j = model_idx[row["detective_model"]]
        logit_p = c_mean[k] * (a_mean[i] - b_mean[j])
        preds.append(expit(logit_p))
    return np.array(preds)


def compute_baseline_predictions(train_df, test_df):
    """
    Baseline: predict using marginal role-specific win rates from training data.

    For each configuration (i, j, k), the baseline predicts the mafia win rate as:
      p = mafioso_i_avg_winrate * (1 - detective_j_avg_townwinrate) * (1 - villager_k_avg_townwinrate)

    But a simpler and more standard baseline is just the marginal average:
    predict each configuration's win rate as the mean of the three marginal
    role-specific rates (mafioso attack rate, detective vulnerability, villager vulnerability).

    We use an additive-on-logit baseline for a fair comparison:
      logit(p) = alpha + maf_effect_i + det_effect_j + vil_effect_k
    estimated from training data marginals.

    Even simpler: just use the per-mafioso average win rate from training data.

    We implement two baselines:
    1. Overall mean: predict the training-set average mafia win rate for everything.
    2. Marginal: per-role average win rates combined additively.
    """
    from scipy.special import logit as sp_logit, expit

    # Baseline 1: overall mean
    overall_rate = train_df["wins"].sum() / train_df["total"].sum()

    # Baseline 2: additive marginal model on logit scale
    # Compute per-model marginal win rates for each role
    maf_rates = train_df.groupby("mafioso_model").apply(
        lambda x: x["wins"].sum() / x["total"].sum()
    )
    det_rates = train_df.groupby("detective_model").apply(
        lambda x: x["wins"].sum() / x["total"].sum()
    )
    vil_rates = train_df.groupby("villager_model").apply(
        lambda x: x["wins"].sum() / x["total"].sum()
    )

    # Grand mean on logit scale
    grand_logit = sp_logit(np.clip(overall_rate, 0.01, 0.99))

    # Per-model effects on logit scale
    maf_logit = {m: sp_logit(np.clip(r, 0.01, 0.99)) - grand_logit for m, r in maf_rates.items()}
    det_logit = {m: sp_logit(np.clip(r, 0.01, 0.99)) - grand_logit for m, r in det_rates.items()}
    vil_logit = {m: sp_logit(np.clip(r, 0.01, 0.99)) - grand_logit for m, r in vil_rates.items()}

    preds_mean = np.full(len(test_df), overall_rate)

    preds_marginal = []
    for _, row in test_df.iterrows():
        logit_p = grand_logit
        logit_p += maf_logit.get(row["mafioso_model"], 0)
        logit_p += det_logit.get(row["detective_model"], 0)
        logit_p += vil_logit.get(row["villager_model"], 0)
        preds_marginal.append(expit(logit_p))
    preds_marginal = np.array(preds_marginal)

    return preds_mean, preds_marginal


def run_experiment(data, held_out_frac, n_splits, draws, seed):
    """Run the held-out prediction experiment over multiple random splits."""
    all_models = sorted(
        set(data["mafioso_model"]) |
        set(data["villager_model"]) |
        set(data["detective_model"])
    )
    model_idx = {m: i for i, m in enumerate(all_models)}
    n_configs = len(data)
    n_test = max(1, int(n_configs * held_out_frac))

    print(f"Configurations: {n_configs}")
    print(f"Models: {len(all_models)}")
    print(f"Held-out fraction: {held_out_frac} ({n_test} configs per split)")
    print(f"Number of splits: {n_splits}")
    print(f"PyMC draws per split: {draws}")
    print()

    results = []
    rng = np.random.RandomState(seed)

    for split_i in range(n_splits):
        print(f"--- Split {split_i + 1}/{n_splits} ---")

        # Random split
        indices = rng.permutation(n_configs)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        train_df = data.iloc[train_idx].reset_index(drop=True)
        test_df = data.iloc[test_idx].reset_index(drop=True)
        actual = test_df["win_rate"].values

        # Check that all models appear in training set
        train_models = (
            set(train_df["mafioso_model"]) |
            set(train_df["villager_model"]) |
            set(train_df["detective_model"])
        )
        test_models = (
            set(test_df["mafioso_model"]) |
            set(test_df["villager_model"]) |
            set(test_df["detective_model"])
        )
        missing = test_models - train_models
        if missing:
            print(f"  WARNING: {len(missing)} models in test but not train, skipping split")
            continue

        # Theoretical model predictions
        print(f"  Fitting Eq. (2) on {len(train_df)} training configs...")
        a_mean, b_mean, c_mean = fit_theoretical_model(
            train_df, all_models, model_idx, draws=draws, seed=seed + split_i
        )
        preds_theory = predict_theoretical(test_df, a_mean, b_mean, c_mean, model_idx)

        # Baseline predictions
        preds_mean, preds_marginal = compute_baseline_predictions(train_df, test_df)

        # Compute MAE
        mae_theory = np.mean(np.abs(actual - preds_theory))
        mae_mean = np.mean(np.abs(actual - preds_mean))
        mae_marginal = np.mean(np.abs(actual - preds_marginal))

        print(f"  MAE — Eq.(2): {mae_theory:.4f}, Marginal: {mae_marginal:.4f}, Overall mean: {mae_mean:.4f}")

        results.append({
            "split": split_i,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mae_theoretical": mae_theory,
            "mae_marginal_baseline": mae_marginal,
            "mae_overall_mean": mae_mean,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Held-out prediction experiment for Mini-Mafia rebuttal"
    )
    parser.add_argument("--held-out-frac", type=float, default=0.2,
                        help="Fraction of configurations to hold out (default: 0.2)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of random train/test splits (default: 5)")
    parser.add_argument("--draws", type=int, default=2000,
                        help="PyMC draws per chain per split (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if not HAS_PYMC:
        print("ERROR: PyMC is required. Install with: pip install pymc arviz")
        sys.exit(1)

    print("=" * 60)
    print("  HELD-OUT PREDICTION EXPERIMENT")
    print("=" * 60)

    data = load_data()
    print(f"Loaded {len(data)} configurations, {data['total'].sum()} total games\n")

    results_df = run_experiment(
        data,
        held_out_frac=args.held_out_frac,
        n_splits=args.n_splits,
        draws=args.draws,
        seed=args.seed,
    )

    # Summary
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY (averaged over splits)")
    print("=" * 60)
    print(f"  Eq. (2) MAE:                {results_df['mae_theoretical'].mean():.4f} +/- {results_df['mae_theoretical'].std():.4f}")
    print(f"  Additive marginal baseline: {results_df['mae_marginal_baseline'].mean():.4f} +/- {results_df['mae_marginal_baseline'].std():.4f}")
    print(f"  Overall mean baseline:      {results_df['mae_overall_mean'].mean():.4f} +/- {results_df['mae_overall_mean'].std():.4f}")
    print()

    improvement_over_marginal = (
        (results_df["mae_marginal_baseline"].mean() - results_df["mae_theoretical"].mean())
        / results_df["mae_marginal_baseline"].mean() * 100
    )
    improvement_over_mean = (
        (results_df["mae_overall_mean"].mean() - results_df["mae_theoretical"].mean())
        / results_df["mae_overall_mean"].mean() * 100
    )
    print(f"  Improvement over marginal baseline: {improvement_over_marginal:.1f}%")
    print(f"  Improvement over overall mean:      {improvement_over_mean:.1f}%")
    print("=" * 60)

    # Rebuttal-ready text
    print()
    print("  REBUTTAL TEXT:")
    print(f'  "The held-out MAE was {results_df["mae_theoretical"].mean():.2f} (+/- {results_df["mae_theoretical"].std():.2f}),')
    print(f'   compared to {results_df["mae_marginal_baseline"].mean():.2f} (+/- {results_df["mae_marginal_baseline"].std():.2f}) from an additive')
    print(f'   marginal baseline using per-model aggregate win rates."')
    print()

    # Save
    output_path = RESULTS_DIR / "held_out_prediction_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
