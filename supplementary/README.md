# Supplementary Material: AutoSI

Implementation code for reproducing the experimental results of
*"Automatic Statistical Test for Rationally Expressible Algorithms by
Selective Inference, with Applications to Feature Selection."*

This archive is anonymized for review.

## Contents

```
autosi/                       The AutoSI library (tracked array type and
                              the automatic selective-inference routine)
experiments/
├── common/
│   ├── environment.py        Thread pinning (1 thread/process) and seeding
│   ├── methods.py            The three feature-selection methods written
│   │                         as ordinary array code (marginal screening,
│   │                         lasso, cross-validated lasso)
│   ├── merge_results.py      CSV loading/merging helpers
│   └── plot_style.py         Shared figure style
├── synthetic/
│   ├── data.py               Synthetic data generation (iid / corr)
│   ├── run_experiment.py     Runner: p-values of all compared methods
│   ├── plot_fpr.py           Figure 2 (type I error rates)
│   └── plot_power.py         Figure 3 (power)
└── realdata/
    ├── datasets.py           Download/cache of the seven UCI datasets
    ├── run_realdata.py       Runner: rejection rates on subsamples
    └── plot_realdata.py      Tables 2 and 4 (rejection rates)
scripts/                      Shell scripts running the paper's settings
```

## Requirements

Python 3.11. Install the dependencies with

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used to produce the reported
numbers.

All commands below are executed from this directory (the supplementary
root); the `autosi` package is imported from here directly, so no
installation step is needed.

## Quick check (a few minutes)

A small smoke test of the whole pipeline (10 p-values, single process):

```bash
python experiments/synthetic/run_experiment.py \
    --methods lasso --data_types iid --iter 10 --out result/smoke.csv
```

The printed summary reports the empirical type I error rate at
alpha = 0.05 over the 10 null p-values.

## Reproducing the synthetic-data experiments (Figures 2 and 3)

```bash
bash scripts/run_synthetic_fpr.sh      # type I error rates (Figure 2)
bash scripts/run_synthetic_power.sh    # power (Figure 3)
bash scripts/make_figures.sh           # render the figures from the CSVs
```

Settings follow the paper: d = 20 features, 3 signal features,
n in {100, 200, 300, 400} for the type I error rate, n = 100 and
Delta in {0.1, 0.2, 0.3, 0.4} for power, 1000 p-values per condition,
alpha = 0.05. The compared methods (proposed, w/o-pp, naive, bonferroni,
data splitting) are all computed in the same run and stored in one CSV row
per trial.

The full setting is computationally demanding — the paper's numbers were
produced on a many-core CPU server with the work split across jobs. Both
runners support splitting by condition (`--methods`, `--data_types`,
`--n_samples`, `--signal`) and by seed range (`--seed_start`, `--seed_end`,
`--seed_offset`); each partial run writes its own CSV, and the plotting
scripts accept a directory and read all CSVs recursively. For a quick trial
on a laptop, pass a smaller count, e.g.
`bash scripts/run_synthetic_fpr.sh 100 8` (100 p-values, 8 workers).

## Reproducing the real-data experiments (Tables 2 and 4)

```bash
bash scripts/run_realdata.sh           # downloads the UCI data on first run
bash scripts/make_figures.sh           # writes realdata_table_<method>.csv
```

The seven regression datasets (Airfoil, Concrete, Energy heating/cooling,
Real estate, Wine red/white) are fetched from the UCI Machine Learning
Repository (CC BY 4.0) and cached under `data/realdata/`. Each trial
subsamples n = 100 observations, standardizes features and response,
estimates the noise variance from OLS residuals, and computes the selective
p-value; 100 p-values are collected per (method, dataset).

## Outputs

- `result/fpr_<method>_<data_type>.pdf` — type I error rate curves
- `result/conditional_power_<method>_<data_type>.pdf` — power curves
- `result/legend_fpr.pdf`, `result/legend_power.pdf` — shared legends
- `result/realdata_table_<method>.csv` — rejection-rate tables, cells
  formatted as `proposed/w-o-pp/bonferroni`
- `result/realdata_summary.csv` — all rejection rates in long format

## Notes

- Every trial is driven by an explicit integer seed (recorded in the
  output CSVs), so all reported numbers are reproducible.
- `si.inference(...)` is the single entry point of AutoSI: it takes the
  user-written algorithm, the contrast eta obtained from one run on the
  observed data, and the noise (co)variance, and returns the selective
  p-value. `inference_mode="over_conditioning"` gives the w/o-pp ablation;
  the naive and Bonferroni p-values are derived from the same result
  object.
