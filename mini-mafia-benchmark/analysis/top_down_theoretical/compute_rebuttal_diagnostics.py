#!/usr/bin/env python3
"""
Compute MCMC diagnostics for the Mini-Mafia rebuttal.

Reports: R-hat range, minimum bulk ESS, minimum tail ESS, maximum MCSE.

Can operate in two modes:
  1. --from-csv   : read the existing mcmc_diagnostics.csv (fast, no sampling)
  2. --resample   : re-run the PyMC model with more chains/samples for
                    tighter diagnostics, then report (slow, requires PyMC)

Usage:
    python compute_rebuttal_diagnostics.py --from-csv
    python compute_rebuttal_diagnostics.py --resample [--chains 4] [--draws 4000] [--tune 2000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "top_down_theoretical"
DB_PATH = Path(__file__).parent.parent.parent / "database" / "mini_mafia.db"
DIAGNOSTICS_CSV = RESULTS_DIR / "mcmc_diagnostics.csv"
REBUTTAL_CSV = RESULTS_DIR / "rebuttal_diagnostics_summary.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def summarise_diagnostics(df: pd.DataFrame) -> dict:
    """Extract the four rebuttal-relevant summary statistics."""
    return {
        "r_hat_min": float(df["r_hat"].min()),
        "r_hat_max": float(df["r_hat"].max()),
        "ess_bulk_min": float(df["ess_bulk"].min()),
        "ess_tail_min": float(df["ess_tail"].min()),
        "mcse_mean_max": float(df["mcse_mean"].max()),
        "mcse_sd_max": float(df["mcse_sd"].max()),
    }


def print_rebuttal_block(stats: dict) -> None:
    """Pretty-print the values needed for the rebuttal placeholders."""
    print()
    print("=" * 60)
    print("  REBUTTAL DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  R-hat range       : [{stats['r_hat_min']:.4f}, {stats['r_hat_max']:.4f}]")
    print(f"  Min bulk ESS      : {stats['ess_bulk_min']:.0f}")
    print(f"  Min tail ESS      : {stats['ess_tail_min']:.0f}")
    print(f"  Max MCSE (mean)   : {stats['mcse_mean_max']:.4f}")
    print(f"  Max MCSE (sd)     : {stats['mcse_sd_max']:.4f}")
    print("=" * 60)

    # Interpretation for the rebuttal text
    print()
    if stats["r_hat_max"] <= 1.01:
        print("  R-hat: GOOD — all values <= 1.01")
    elif stats["r_hat_max"] <= 1.05:
        print(f"  R-hat: ACCEPTABLE — max {stats['r_hat_max']:.4f} (< 1.05)")
    else:
        print(f"  R-hat: WARNING — max {stats['r_hat_max']:.4f} (> 1.05, convergence issues)")

    if stats["ess_bulk_min"] >= 400:
        print(f"  Bulk ESS: GOOD — minimum {stats['ess_bulk_min']:.0f} (>= 400)")
    elif stats["ess_bulk_min"] >= 100:
        print(f"  Bulk ESS: ACCEPTABLE — minimum {stats['ess_bulk_min']:.0f} (>= 100 but < 400)")
    else:
        print(f"  Bulk ESS: WARNING — minimum {stats['ess_bulk_min']:.0f} (< 100)")

    if stats["ess_tail_min"] >= 400:
        print(f"  Tail ESS: GOOD — minimum {stats['ess_tail_min']:.0f} (>= 400)")
    elif stats["ess_tail_min"] >= 100:
        print(f"  Tail ESS: ACCEPTABLE — minimum {stats['ess_tail_min']:.0f} (>= 100 but < 400)")
    else:
        print(f"  Tail ESS: WARNING — minimum {stats['ess_tail_min']:.0f} (< 100)")
    print()


# ── mode 1: from existing CSV ───────────────────────────────────────────────

def from_csv() -> dict:
    """Read diagnostics from the pre-computed CSV."""
    if not DIAGNOSTICS_CSV.exists():
        print(f"ERROR: {DIAGNOSTICS_CSV} not found. Run --resample or generate it first.")
        sys.exit(1)

    df = pd.read_csv(DIAGNOSTICS_CSV, index_col=0)
    print(f"Loaded {len(df)} parameter rows from {DIAGNOSTICS_CSV}")
    return summarise_diagnostics(df)


# ── mode 2: resample ────────────────────────────────────────────────────────

def resample(chains: int, draws: int, tune: int) -> dict:
    """Re-run the PyMC model with the given sampling settings."""
    try:
        import pymc as pm
        import arviz as az
        import sqlite3
    except ImportError as e:
        print(f"ERROR: missing dependency — {e}")
        print("Install with: pip install pymc arviz")
        sys.exit(1)

    # ── load data (reuse logic from scores_theoretical_model.py) ─────────
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

    all_models = sorted(
        set(grouped["mafioso_model"]) |
        set(grouped["villager_model"]) |
        set(grouped["detective_model"])
    )
    model_idx = {m: i for i, m in enumerate(all_models)}
    n_models = len(all_models)

    maf_ix = [model_idx[m] for m in grouped["mafioso_model"]]
    vil_ix = [model_idx[m] for m in grouped["villager_model"]]
    det_ix = [model_idx[m] for m in grouped["detective_model"]]
    wins = grouped["wins"].values
    totals = grouped["total"].values

    print(f"Data: {len(grouped)} configurations, {totals.sum()} total games, {n_models} models")

    # ── build & sample ───────────────────────────────────────────────────
    with pm.Model() as model:
        a = pm.Normal("a", mu=0, sigma=2, shape=n_models)
        b = pm.Normal("b", mu=0, sigma=2, shape=n_models)
        c = pm.Normal("c", mu=0, sigma=2, shape=n_models)

        # identifiability: sum(c) > 0
        pm.Potential("c_sum_positive",
                     pm.math.switch(pm.math.sum(c) > 0, 0, -np.inf))

        logit_p = c[vil_ix] * (a[maf_ix] - b[det_ix])
        pm.Binomial("y", n=totals, logit_p=logit_p, observed=wins)

        print(f"Sampling: {chains} chains x {draws} draws (tune={tune}) ...")
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=0.95,
            random_seed=42,
            progressbar=True,
        )

    # ── diagnostics ──────────────────────────────────────────────────────
    diag = az.summary(trace, var_names=["a", "b", "c"], round_to=4)
    diag.to_csv(DIAGNOSTICS_CSV)
    print(f"Diagnostics saved to {DIAGNOSTICS_CSV}")

    return summarise_diagnostics(diag)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute MCMC diagnostics for the Mini-Mafia rebuttal"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-csv", action="store_true",
                      help="Read existing mcmc_diagnostics.csv")
    mode.add_argument("--resample", action="store_true",
                      help="Re-run PyMC sampling with better settings")
    parser.add_argument("--chains", type=int, default=4,
                        help="Number of MCMC chains (default: 4)")
    parser.add_argument("--draws", type=int, default=4000,
                        help="Draws per chain after tuning (default: 4000)")
    parser.add_argument("--tune", type=int, default=2000,
                        help="Tuning steps per chain (default: 2000)")

    args = parser.parse_args()

    if args.from_csv:
        stats = from_csv()
    else:
        stats = resample(args.chains, args.draws, args.tune)

    print_rebuttal_block(stats)

    # Save summary to a small CSV for easy reference
    summary_df = pd.DataFrame([stats])
    summary_df.to_csv(REBUTTAL_CSV, index=False)
    print(f"Summary saved to {REBUTTAL_CSV}")


if __name__ == "__main__":
    main()
