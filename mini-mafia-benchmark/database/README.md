# Mini-Mafia Databases

This folder contains the SQLite databases used by the Mini-Mafia benchmarks. All
files share the same schema defined in `schema.sql`, but each database captures a
different experiment setup.

## Databases

- `mini_mafia.db`: Legacy 2-round benchmark used for baseline comparisons and the
  original "deceive/detect/disclose" capability tracking via the `benchmark`
table. The `run_mini_mafia_batch.py` script used this file before the
round-specific variants existed.

- `mini_mafia_round4.db`: Four-discussion-round experiment where all roles use
  GPT-5 Mini. Scripts such as `win_rates_round4_plot.py` and
  `run_mini_mafia_batch_roundn.py` read from or write to this database.

- `mini_mafia_round8.db`: Eight-discussion-round experiment created for the
  extended benchmark. The schema is identical to `mini_mafia_round4.db`, but it
  stores the longer games launched via `run_mini_mafia_batch_roundn.py`.

- `mini_mafia_short_prompt.db`: Short-prompt deceive experiment database. The
  `run_short_prompt_experiment.py` helper routes batches here so the alternative
  prompt data set stays isolated from the round-based experiments.

Each `.db` file can be inspected with `sqlite3 <path> '.schema'` (structure) or
`sqlite3 <path> '.tables'` (table listing). The shared helper `database/db_utils.py`
can connect to any of them by passing the desired `db_path` when creating
`MiniMafiaDB` instances.
