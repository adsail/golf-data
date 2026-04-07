# Masters regression — what this models

## Outcome and unit of observation

- **Outcome (Y):** Masters finish position (numeric rank from `fin_text` in Data Golf’s pre-tournament archive). Lower is better.
- **Unit:** One row per **player** per **Masters season** (`season_year`), for training years configured in `MASTERS_ARCHIVE` in `[masters_regression.py](masters_regression.py)`.

## Data sources

- **Historical raw event list:** `[/historical-raw-data/event-list](https://feeds.datagolf.com/historical-raw-data/event-list)` (`tour=pga`) to find PGA events and Masters dates.
- **Pre-tournament archive:** `[/preds/pre-tournament-archive](https://feeds.datagolf.com/preds/pre-tournament-archive)` for Masters labels and for every PGA event in the **six calendar months before** each Masters (excluding Masters weeks).
- **Skill-style snapshots:** `[/preds/skill-ratings](https://feeds.datagolf.com/preds/skill-ratings)`, `[/preds/approach-skill](https://feeds.datagolf.com/preds/approach-skill)`, `[/preds/player-decompositions](https://feeds.datagolf.com/preds/player-decompositions)` — fetched on every run for **`age_at_masters`** and, when enabled, for optional skill columns. Skill metrics are **current** tour-wide / Augusta-fit estimates, not historically time-stamped to each training year.

Set `DATAGOLF_KEY` in the environment; see [Data Golf API access](https://datagolf.com/api-access).

## Tournament block (always in the model)

These **nine** predictors are always included (plus the usual caveats on imputation below):

| Column                             | Meaning                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `masters_prior_starts`             | Prior Masters with a numeric finish                                     |
| `masters_prior_decay_mean_finish`  | Decay-weighted mean prior finish, weight ∝ exp(−λ·years before Masters); λ = `--masters-decay-lambda` (default 0.25) |
| `masters_prior_best_recent`        | Best prior finish among Masters in the last `--masters-recent-best-years` calendar years before that season (default 8) |
| `pga6m_starts`                     | Count of PGA events in the prior ~6 months with a recorded numeric finish |
| `pga6m_mean_finish`                | Mean finish across those events                                         |
| `pga6m_top25_rate`                 | Share of those events finishing top 25                                  |
| `pga6m_made_cut_rate`              | Share of rounds with a made-cut numeric finish                          |
| `pga6m_insufficient`               | 1 if `pga6m_starts` is below `--min-pga6m-starts` (default 3); form metrics for those rows are set to pessimistic constants instead of tour-like medians |
| `age_at_masters`                   | From skill-ratings `age`: on training rows, `age − (pga_ref_year − season_year)` with `--pga-ref-year` (default 2026); for the **current** 2026 projection row, `age` is used as-is |

**Imputation:** Prior Masters columns and `age_at_masters` use **within-season medians**, then global medians. PGA 6m metrics use the same **after** a first pass, then rows with fewer than `--min-pga6m-starts` PGA finishes in the window are **overwritten** with pessimistic placeholders (rough field tail for mean finish, low top-25 rate, modest made-cut rate) so the model does not treat sparse schedules as “median tour form.”

**2026 ranked table:** By default, players with `pga6m_starts` below the minimum are **listed and excluded** from the printed prediction ranking; pass `--no-skip-predict-below-min-pga6m` to score the full field anyway.

## Skill candidates (optional)

When you use `--skill-forward-select` or `--include-skill-features`, the script merges extra columns. Forward selection considers (in order) whichever of these exist in the merged frame:

`sg_app`, `sg_putt`, `sg_arg`, `sg_ott`, `driving_dist`, `driving_acc` (FIR-style vs tour), `gir_pct` (if the skill feed exposes `gir` / `gir_pct` / `gir_rate`), `par3_approach_proxy` (mean of short fairway approach SG buckets), `par5_long_approach_proxy` (200+ yd fairway SG), `course_history`, `timing_adjustment`, `short_game_tight_lie_proxy`.

## Forward selection (`--skill-forward-select`)

**Gate A (in sample):** For each candidate not yet selected and not **LOYO-failed**, fit nested OLS: tournament + current selection vs tournament + selection + candidate. Require **partial F** with **p < `--entry-alpha`** (default 0.05). Among passing candidates, try **smallest p first**.

**Gate B (LOYO):** **Leave-one-`season_year`-out:** for each Masters year, fit the reduced and full models on all other years and compute RMSE on the held-out year. Admit the candidate only if **mean LOYO RMSE(full) ≤ mean LOYO RMSE(reduced) − `--loyo-epsilon`** (default ε = 0).

If the best significant candidate fails LOYO, it is **blacklisted** for the rest of the run; the next-smallest-p candidate is tried in the same round. If none pass LOYO, selection **stops**. After a successful add, the loop **repeats** with an updated selected set (candidates with p ≥ α earlier may become significant later).

**Important:** Gate A uses **classical** OLS F-tests. The **final** printed regression summary uses **cluster-robust standard errors by `dg_id`** when possible. The block **partial F-tests** printed at the end are also classical nested OLS (same logic as Gate A), not cluster-robust Wald tests.

## Fixed skill bundle (`--include-skill-features`)

Appends a **fixed** set of six composites (`sg_t2g`, `sg_app`, `par5_long_approach_proxy`, `course_history`, `short_game_tight_lie_proxy`, `length_proxy`) with **no** forward selection or LOYO.

If both `--include-skill-features` and `--skill-forward-select` are set, **forward selection wins**.

## CLI reference

```bash
export DATAGOLF_KEY=your_key
pip install -r requirements.txt

# Tournament-only
python masters_regression.py

# Forward-selected skills
python masters_regression.py --skill-forward-select --entry-alpha 0.05 --loyo-epsilon 0

# Fixed skill bundle
python masters_regression.py --include-skill-features

# Restrict to players with a top-25 in the prior 6 months on PGA
python masters_regression.py --require-top25-in-6m

# Tune prior-Masters decay, recent-best window, 6m minimum starts, age reference year
python masters_regression.py --masters-decay-lambda 0.2 --masters-recent-best-years 10 \
  --min-pga6m-starts 4 --pga-ref-year 2026

# Include everyone in 2026 rankings despite low 6m starts
python masters_regression.py --no-skip-predict-below-min-pga6m
```

## Reading the output

- **Forward selection:** stderr lines `Step N ADMIT/reject (LOYO)` with **F**, **df**, **p**, **ΔR²**, and **LOYO RMSE** (reduced vs full).
- **Skill selection summary** (stdout, after forward selection): consolidated lists:
  - **Tournament block** — the fixed tournament + age + low-sample-flag predictors always in the model.
  - **Skill metrics admitted** — columns that entered via forward selection.
  - **Unavailable** — candidates in the ordered pool that never had usable values in the merged training frame.
  - **Passed Gate A but failed LOYO** — significant in-sample once, but mean leave-one-year-out RMSE did not improve enough.
  - **Not significant vs final model** — present in data, not LOYO-failed: end-state nested OLS vs **tournament + final selected skills** has **p ≥ entry-alpha** (with **p**, **F**, **ΔR²** printed).
  - **Insufficient data** — end-state nested test could not be fit for that column.
  - **Significant vs final but not in model** — rare; **p < alpha** vs the final reduced set but the stepwise path did not retain the variable.
- **Final OLS:** coefficient table with cluster SE when available.
- **Partial F-tests:** joint test of dropping a **block** (prior Masters, PGA 6m + `pga6m_insufficient`, `age_at_masters`, or selected skills).

## Limitations

- **`age_at_masters` on training rows** uses a **back-cast** from today’s API `age` and `season_year`; it is an approximation, not a true age-at-event time series.
- Skill ratings (when used) are **snapshots**, not necessarily aligned to each historical Masters week — training rows mix past outcomes with **present** skill estimates when skills are used.
- **Stepwise + LOYO** reduces overfitting somewhat but does **not** give fully valid post-selection inference; treat as **exploratory**.
- Some API tiers cannot access certain historical endpoints; this pipeline relies on **event-list + pre-tournament-archive** for tournament history.

