#!/usr/bin/env python3
"""Run the short-prompt deceive experiment batches sequentially."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from run_mini_mafia_batch import run_batch

# Canonical model presets used for both background roles and mafioso variants.
MODEL_PRESETS: Dict[str, Dict] = {
    'gpt-5-mini': {
        'type': 'openai',
        'model': 'gpt-5-mini',
        'temperature': 0.7,
        'reasoning_effort': 'minimal'
    },
    'gpt-4.1-mini': {
        'type': 'openai',
        'model': 'gpt-4.1-mini',
        'temperature': 0.7
    },
    'grok-3-mini': {
        'type': 'xai',
        'model': 'grok-3-mini',
        'temperature': 0.7
    },
    'mistral': {
        'type': 'local',
        'model': 'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf',
        'temperature': 0.7
    },
    'deepseek': {
        'type': 'deepseek',
        'model': 'deepseek-chat',
        'temperature': 0.7
    },
    'claude-sonnet': {
        'type': 'anthropic',
        'model': 'claude-sonnet-4-20250514',
        'temperature': 0.7,
        'use_cache': True
    },
    'claude-opus': {
        'type': 'anthropic',
        'model': 'claude-opus-4-1-20250805',
        'temperature': 0.7,
        'use_cache': True
    },
    'llama': {
        'type': 'local',
        'model': 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        'temperature': 0.7
    },
    'qwen': {
        'type': 'local',
        'model': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
        'temperature': 0.7
    }
}

BACKGROUND_SEQUENCE = ['gpt-4.1-mini']
TARGET_MODELS = ['qwen']

DEFAULT_DB_PATH = Path(__file__).resolve().parent / 'database' / 'mini_mafia_short_prompt.db'


def build_model_config(background: str, target: str) -> Dict[str, Dict]:
    """Return the role->model configuration for a background/target pair."""
    try:
        bg_preset = MODEL_PRESETS[background]
        target_preset = MODEL_PRESETS[target]
    except KeyError as exc:
        raise ValueError(f"Unknown model preset: {exc.args[0]}") from exc

    # Use deepcopy to avoid accidental mutation across games.
    config = {
        'detective': deepcopy(bg_preset),
        'villager': deepcopy(bg_preset),
        'mafioso': deepcopy(target_preset)
    }
    return config


def generate_schedule() -> List[Dict]:
    """Create the ordered list of experiment configurations."""
    schedule = []
    for background in BACKGROUND_SEQUENCE:
        for target in TARGET_MODELS:
            schedule.append({
                'background': background,
                'target': target,
                'model_configs': build_model_config(background, target)
            })
    return schedule


def export_schedule(schedule: List[Dict], export_path: Path) -> None:
    export_path.write_text(
        json.dumps([
            {
                'index': idx,
                'background': entry['background'],
                'target': entry['target'],
                'model_configs': entry['model_configs']
            }
            for idx, entry in enumerate(schedule, start=1)
        ], indent=2
        ) + '\n'
    )
    print(f"Exported schedule to {export_path}")


def run_short_prompt_experiment(games_per_batch: int, db_path: Path, start_index: int,
                                limit: int | None, debug_prompts: bool, dry_run: bool) -> None:
    schedule = generate_schedule()
    total = len(schedule)

    if start_index < 1 or start_index > total:
        raise ValueError(f"start_index must be between 1 and {total}")

    plan = schedule[start_index - 1:]
    if limit is not None:
        plan = plan[:limit]

    print(f"Running short-prompt deceive experiment batches")
    print(f"Total configurations available: {total}")
    print(f"Starting at index {start_index}; executing {len(plan)} combinations")
    print(f"Database: {db_path}")

    for offset, entry in enumerate(plan, start=start_index):
        background = entry['background']
        target = entry['target']
        print("-" * 80)
        print(f"[{offset}/{total}] Background: {background} | Mafioso: {target}")
        if dry_run:
            continue

        batch_id = run_batch(
            n_games=games_per_batch,
            debug_prompts=debug_prompts,
            model_configs=entry['model_configs'],
            db_path=str(db_path)
        )
        print(f"Finished batch {batch_id} for background={background}, mafioso={target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the short-prompt deceive experiment sequence.')
    parser.add_argument('games_per_batch', type=int, nargs='?', default=100,
                        help='Number of games to run per batch (default: 100)')
    parser.add_argument('--db-path', type=Path, default=DEFAULT_DB_PATH,
                        help=f"SQLite DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument('--start-index', type=int, default=1,
                        help='1-based index of schedule entry to start from (default: 1)')
    parser.add_argument('--limit', type=int,
                        help='Limit number of schedule entries to execute')
    parser.add_argument('--debug-prompts', action='store_true', help='Show LLM prompts during games')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print schedule information without running games')
    parser.add_argument('--export-schedule', type=Path,
                        help='Write the full configuration schedule to the given JSON file')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = generate_schedule()

    if args.export_schedule:
        export_schedule(schedule, args.export_schedule)
        if args.dry_run and args.limit is None:
            # If user only wanted the export, exit early.
            return

    run_short_prompt_experiment(
        games_per_batch=args.games_per_batch,
        db_path=args.db_path,
        start_index=args.start_index,
        limit=args.limit,
        debug_prompts=args.debug_prompts,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
