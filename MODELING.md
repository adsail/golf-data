# Masters regression — period-accurate form + Augusta modifiers

## Outcome and unit of observation

- **Outcome (Y):** Masters finish position (numeric rank from `fin_text` in Data Golf pre-tournament archive). Lower is better.
- **Unit:** One row per **player** per **Masters season** (`season_year`) for training seasons in `MASTERS_ARCHIVE`.

## Data sources

- **Historical event list:** `[/historical-raw-data/event-list](https://feeds.datagolf.com/historical-raw-data/event-list)` (`tour=pga`) to identify events and dates.
- **Pre-tournament archive (only):** `[/preds/pre-tournament-archive](https://feeds.datagolf.com/preds/pre-tournament-archive)` for:
  - Masters labels (`finish_rank`) in each training year.
  - Prior-6-month PGA events for each year's form features.
  - Prior Masters history features.

No current snapshot skill endpoints are used in the model path.

## Feature design

The model uses:

1. **Current form engine (period-accurate):** from PGA events in the six months before that year's Masters.
2. **Augusta modifier:** prior Masters-specific history.

### Tournament block (always included)


| Column                            | Meaning                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| `masters_prior_starts`            | Prior Masters starts with numeric finish                                                        |
| `masters_prior_decay_mean_finish` | Decay-weighted prior finish (`exp(-lambda * years_lag)`), `--masters-decay-lambda`              |
| `masters_prior_best_recent`       | Best prior Masters finish in recent window, `--masters-recent-best-years`                       |
| `masters_prior_top10_count`       | Count of prior Masters top-10 finishes                                                          |
| `pga6m_starts`                    | Number of PGA starts in prior 6 months with numeric finish                                      |
| `pga6m_mean_finish`               | Mean finish in those starts                                                                     |
| `pga6m_top25_rate`                | Top-25 share in those starts                                                                    |
| `pga6m_made_cut_rate`             | Made-cut share in those starts                                                                  |
| `pga6m_sg_app_mean`               | Mean SG:Approach over prior-6-month event rows (alias-matched from archive columns)             |
| `pga6m_sg_ott_mean`               | Mean SG:Off-the-Tee over prior-6-month event rows                                               |
| `pga6m_sg_putt_mean`              | Mean SG:Putting over prior-6-month event rows                                                   |
| `pga6m_insufficient`              | 1 if `pga6m_starts < --min-pga6m-starts`; low-sample rows receive pessimistic form replacements |


### Imputation and low-sample behavior

- Numeric features are imputed by season median, then global median.
- For players below `--min-pga6m-starts`:
  - `pga6m_mean_finish`, `pga6m_top25_rate`, `pga6m_made_cut_rate` are overwritten with pessimistic constants.
  - `pga6m_insufficient=1` is set.

### 2026 prediction pipeline

- Uses the **same** 6-month pre-Masters archive feature engineering.
- Applies the same prior Masters modifier logic.
- By default, excludes players below `--min-pga6m-starts` from ranked output (`--no-skip-predict-below-min-pga6m` to include).

## CLI reference

```bash
export DATAGOLF_KEY=your_key
pip install -r requirements.txt

python masters_regression.py
python masters_regression.py --masters-decay-lambda 0.2 --masters-recent-best-years 10
python masters_regression.py --min-pga6m-starts 4
python masters_regression.py --no-skip-predict-below-min-pga6m
python masters_regression.py --output-csv out/masters_2026.csv
```

## Notes and limitations

- SG columns in archives may vary by feed naming; the script uses alias and token matching, then aggregates whichever SG columns are present.
- If an SG component is missing in the archive feed for some events/seasons, that component will be sparsely populated and mostly median-imputed.
- Statistical inference is exploratory (small sample across Masters years); partial F-tests are classical nested OLS tests.

