---
name: Mini-Mafia Project Overview
description: ICML 2026 paper benchmarking LLM social intelligence with a simplified Mafia game; theoretical model, Bayesian inference, 14k games, 10 models + humans
type: project
---

# Paper: "Deceive, Detect, and Disclose: Large Language Models Play Mini-Mafia"

Submitted to ICML 2026 (anonymous submission). Repo at `/Users/davicosta/Desktop/projects/llm-mafia-game`.

## Core Idea
Mini-Mafia is a 4-player variant: 1 mafioso, 1 detective, 2 villagers. Night phase is fixed (mafioso kills a villager, detective investigates the mafioso). Only 3 agents survive to the day phase (2 discussion rounds + blind vote). This isolates 3 capabilities: **deceive** (mafioso), **detect** (villager), **disclose** (detective).

## Key Analytic Result
**logit(p_ijk) = v_k × (m_i − d_j)**
- m_i = deception capability of mafioso i
- d_j = disclosure capability of detective j
- v_k = detection sensitivity of villager k
- Gauged by: E[m_i]=0, E[v_k]=1

Derived from 3 axioms: (1) outcome depends on m-d gap, (2) p=1/2 when m=d, (3) p=1/2 when v=0. Has Fermi-Dirac form from statistical physics.

## Empirical Setup
- 10 models: Claude Opus 4.1, Claude Sonnet 4, DeepSeek V3.1, Gemini 2.5 Flash Lite, GPT-4.1 Mini, GPT-5 Mini, Grok 3 Mini, Llama 3.1 8B, Mistral 7B, Qwen2.5 7B
- 140 targeted (i,j,k) configs, 100 games each → 14,000 unique games
- Bayesian inference via PyMC + NUTS (2000 samples, 2 chains). Max R-hat=1.0117.
- 80 human baseline games (from data science summer school web interface)

## Key Results
- Grok 3 Mini = best detector (85.7% accuracy), Claude Sonnet 4 = worst (50.6%, near random)
- GPT-5 Mini + Grok 3 Mini = best disclosers (disclose immediately in >93% of games)
- Llama 3.1 8B = worst discloser (never discloses role in 69.9% of games)
- Humans significantly outperform all LLMs at detection; LLMs excel at persuasion
- Smaller models often outperform larger ones (size ≠ social intelligence)

## Serendipitous Results
- **Name bias**: Bob (55.96%) > Alice (55.55%) > Charlie (54.16%) > Diana (53.76%) win rates; male names slightly favored
- **Last-speaker advantage**: detectives gain +7.10pp when speaking last; mafiosos gain +6.04pp
- Name bias modeled as additive logit shift β_n

## Architecture
- `mini-mafia-benchmark/mini_mafia.py` — game engine (wraps `src/`)
- `experiments/run_mini_mafia_batch.py` — runs tournaments, stores in SQLite
- `database/mini_mafia.db` — main 14k-game DB (schema: games, players, game_players, benchmark)
- `analysis/top_down_theoretical/scores_theoretical_model.py` — PyMC Bayesian inference
- `analysis/bottom_up_background/` — independent z-score validation (5 backgrounds × 10 models × 100 games)
- `web/web_interface.py` — Flask app for human data collection
- `article/main.tex` — LaTeX paper source

## Validation
Two independent approaches agree: top-down theoretical model (logit form, 14k games) and bottom-up background z-score analysis (15k games across 5 fixed backgrounds). Pearson r~0.5 for prompt ablation (capability rankings are prompt-conditioned, not absolute).

**Why:** Demonstrates multi-agent LLM dialogue can be reduced to compact latent parameters with full analytic tractability — a first for a complete 3-agent interaction loop.
**How to apply:** When suggesting experiments or extensions, note the model's constraint structure (gauge symmetries, axiom derivation) and that rankings are prompt-conditioned.
