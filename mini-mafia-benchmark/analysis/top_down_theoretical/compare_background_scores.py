#!/usr/bin/env python3
"""Compare top-down latent effects with bottom-up background z-scores.

This is an internal-consistency analysis rather than independent construct
validation: both score families ultimately use outcomes from the same game
protocol. The output is cited in the AAMAS appendix with that limitation.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
BOTTOM_UP = ROOT / "results" / "bottom_up_background" / "results_table_zscores.csv"
TOP_DOWN = ROOT / "results" / "top_down_theoretical" / "scores_theoretical.csv"
OUTPUT = ROOT / "results" / "top_down_theoretical" / "background_score_correlations.csv"


def main() -> None:
    bottom_up = pd.read_csv(BOTTOM_UP)
    top_down = pd.read_csv(TOP_DOWN)
    merged = bottom_up.merge(top_down, on="Model", validate="one_to_one")

    records = []
    for dimension in ("Deceive", "Detect", "Disclose"):
        pearson = pearsonr(merged[f"{dimension}_Z"], merged[dimension])
        spearman = spearmanr(merged[f"{dimension}_Z"], merged[dimension])
        records.append(
            {
                "dimension": dimension.lower(),
                "n_models": len(merged),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
            }
        )

    result = pd.DataFrame(records)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
