"""Real-data experiments: selective inference on subsamples of UCI datasets.

For each (dataset, subsample size n) pair, every trial performs
  1. subsample n rows without replacement
  2. standardize the columns of X (dropping constant columns) and y; the
     noise variance sigma^2 is estimated from the OLS residuals
     (degrees of freedom n - p - 1, subtracting one for the intercept)
  3. run the algorithm (response centering + feature selection) to get eta
  4. compute the selective p-value via si.inference (this interval is timed)
Because the true signal in real data is unknown, the rejection rate
P(p < alpha) is reported instead of power.

Two execution modes (same scheme as the synthetic run_experiment.py):
  fixed:    --iter (or --seed_start/--seed_end) fixes the number of trials
  dynamic:  --target_pvalues N keeps adding trials until N successful
            trials (p-values) are collected, automatically replacing trials
            that failed (e.g., empty selected set)

Intercept handling: the lasso implementation carries no intercept, so in
addition to the column centering of X the response is centered INSIDE the
algorithm (equivalent to an unpenalized intercept). Centering is a linear
operation on the data, so autosi tracks it like any other operation and the
inference remains valid.

Usage:
  python experiments/realdata/datasets.py                          # once
  python experiments/realdata/run_realdata.py --iter 10 --datasets wine_red
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import before numpy/BLAS so that every process is pinned to one thread
from experiments.common.environment import set_env, set_seed

import time
import argparse
import traceback
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import autosi as si
from experiments.common import methods as methods_lib
from experiments.common.methods import marginal_screening, lasso, cv_r2_lasso
from experiments.realdata.datasets import DATASET_NAMES, load_dataset

METHODS = ["marginal_screening", "lasso", "cv_r2_lasso"]


def _subsample(dataset, n_samples, seed):
    """Subsample n rows from the dataset and standardize."""
    X_full, y_full = load_dataset(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_full), size=n_samples, replace=False)
    X, y = X_full[idx], y_full[idx]

    # Column standardization; columns constant within the subsample are
    # dropped (prevents a singular X^T X).
    std = X.std(axis=0)
    keep = std > 1e-12
    X = (X[:, keep] - X[:, keep].mean(axis=0)) / std[keep]

    # Standardize y as well. On the raw scale the convergence tolerance of
    # coordinate descent would be too strict relative to the coefficient
    # scale. The test of H0: eta^T mu = 0 is invariant to positive scaling
    # of y, so rejection rates are unaffected.
    y = (y - y.mean()) / y.std()
    return X, y


def _estimate_sigma2(X, y):
    """Estimate the noise variance from the OLS residuals (one degree of
    freedom subtracted for the intercept)."""
    n, p = X.shape
    y_c = y - y.mean()
    beta = np.linalg.lstsq(X, y_c, rcond=None)[0]
    resid = y_c - X @ beta
    return float(resid @ resid / (n - p - 1))


def _build_algorithm(method, X, n, p, seed, cfg):
    """Build the algorithm: response centering + feature selection.

    The tested feature is drawn from the selected set with the random
    number pick_u (same rule as the synthetic experiments; pick_u is
    injected into cfg by run_single).
    """
    if method == "marginal_screening":
        base = partial(marginal_screening, X=X, k=cfg["ms_k"],
                       pick_u=cfg["pick_u"])
    elif method == "lasso":
        base = partial(
            lasso, X=X, n_samples=n, n_features=p,
            alpha=cfg["alpha"], max_iter=cfg["max_iter"], tol=cfg["tol"],
            pick_u=cfg["pick_u"],
        )
    elif method == "cv_r2_lasso":
        rng = np.random.default_rng(seed)
        fold_indices = np.array_split(rng.permutation(n), cfg["K"])
        lambda_grid = np.logspace(cfg["lam_lo"], cfg["lam_hi"], cfg["n_lambda"])
        base = partial(
            cv_r2_lasso, X=X, n_samples=n, n_features=p,
            fold_indices=fold_indices, lambda_grid=lambda_grid,
            K=cfg["K"], max_iter=cfg["max_iter"], tol=cfg["tol"],
            pick_u=cfg["pick_u"],
        )
    else:
        raise ValueError(f"unknown method: {method}")

    def algorithm(y):
        # Center y in place of an intercept before selecting features.
        # X is column-centered, so eta^T 1 = 0 and the test statistic is
        # centering-invariant.
        return base(y - si.sum(y) / n)

    return algorithm


def run_single(task):
    """One trial: subsample -> obtain eta -> compute p-values (timed)."""
    method, dataset, n_samples, seed, cfg = task
    set_env(parallel=True)
    set_seed(seed)
    try:
        X_np, y_np = _subsample(dataset, n_samples, seed)
        n, p = X_np.shape
        sigma2 = _estimate_sigma2(X_np, y_np)

        X = si.siarray(X_np)
        y = si.siarray(y_np)
        # Random number for drawing the tested feature (separated from the
        # subsampling seed)
        pick_u = float(np.random.default_rng([seed, 987654321]).uniform())
        cfg = dict(cfg, pick_u=pick_u)
        algorithm = _build_algorithm(method, X, n, p, seed, cfg)

        # Active set on the observed data (lasso variants only); reset it so
        # that no value from a previous task leaks.
        methods_lib.last_active_set = None
        eta = algorithm(y)
        active_set = methods_lib.last_active_set
        # A trial with an empty selection has nothing to test; skip it.
        if active_set is not None and len(active_set) == 0:
            raise RuntimeError("empty selected feature set")
        selected_index = int(methods_lib.last_picked)

        oc_p = si.inference(eta=eta, prob_vec=y, var=sigma2, algorithm=algorithm,
                            inference_mode="over_conditioning").p_value
        start = time.time()
        result = si.inference(eta=eta, prob_vec=y, var=sigma2, algorithm=algorithm)
        elapsed = time.time() - start

        naive_p = result.naive_p_value()
        # Number of comparisons for Bonferroni = 2^p (log is passed)
        bonferroni_p = result.bonferroni_p_value(log_num_comparisons=p * np.log(2))

        return {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "p_value": float(result.p_value),
            "oc_p_value": float(oc_p),
            "naive_p_value": float(naive_p),
            "bonferroni_p_value": float(bonferroni_p),
            "time": elapsed,
            "n_samples": n_samples,
            "n_features": p,
            "sigma2": sigma2,
            # Recorded so each CSV is self-describing about its settings
            "K": cfg["K"],
            "n_lambda": cfg["n_lambda"],
            "lam_lo": cfg["lam_lo"],
            "lam_hi": cfg["lam_hi"],
            "selected_index": selected_index,
            # Active set ("|"-separated; empty for marginal screening)
            "active_set": "" if active_set is None else "|".join(map(str, active_set)),
            "active_size": None if active_set is None else len(active_set),
        }
    except Exception:
        # A single failed trial does not stop the whole run.
        print(f"[skip] method={method} dataset={dataset} n={n_samples} "
              f"seed={seed} failed (continuing)", file=sys.stderr)
        traceback.print_exc()
        return None


def run_dynamic(method, dataset, n_samples, cfg, target, seed_start,
                seed_offset, workers, max_trials):
    """Keep adding trials until `target` successful trials are collected.

    Trials with an empty selected set or numerical failures are skipped, so
    a fixed seed range would yield an uneven number of p-values across
    conditions. The required number of additional trials is estimated from
    the observed success rate and submitted in waves. One trial yields the
    p-values of all methods (proposed / w/o-pp / bonferroni) at once, so
    only the number of successful trials needs tracking.

    Waves are rounded up to a multiple of the worker count so that all
    cores stay busy. Seeds are consumed sequentially from seed_start; shift
    it per job when splitting work across machines.
    """
    records = []
    n_failed = 0
    n_launched = 0
    next_seed = seed_start + seed_offset

    with ProcessPoolExecutor(max_workers=workers) as ex:
        wave = 0
        while len(records) < target:
            # Optimistically smoothed success rate (1.0 before any trial)
            n_done = len(records) + n_failed
            rate = (len(records) + 1) / (n_done + 1)
            need = (target - len(records)) / rate
            n_wave = min(int(np.ceil(need / workers)) * workers,
                         max_trials - n_launched)
            if n_wave <= 0:
                print(f"warning: reached max_trials={max_trials}; short of "
                      f"p-values ({len(records)} / target {target})",
                      file=sys.stderr)
                break
            wave += 1
            tasks = [(method, dataset, n_samples, s, cfg)
                     for s in range(next_seed, next_seed + n_wave)]
            next_seed += n_wave
            n_launched += n_wave
            futures = [ex.submit(run_single, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{method}/{dataset} wave{wave}"):
                try:
                    rec = fut.result()
                except Exception:
                    traceback.print_exc()
                    rec = None
                if rec is None:
                    n_failed += 1
                else:
                    records.append(rec)

    print(f"{method}/{dataset} n={n_samples}: {n_launched} trials for "
          f"{len(records)} p-values (target {target})")
    return records, n_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=1000,
                        help="number of trials per (dataset, n)")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel workers")
    parser.add_argument("--methods", type=str, default="lasso",
                        help="comma-separated method names")
    parser.add_argument("--datasets", type=str, default=",".join(DATASET_NAMES),
                        help="comma-separated dataset names (default: all)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="subsample size n")
    parser.add_argument("--seed_start", type=int, default=None)
    parser.add_argument("--seed_end", type=int, default=None)
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="constant added to every seed (vary it per "
                             "method to keep the subsamples independent)")
    # Dynamic mode: automatically replace failed trials so every condition
    # ends up with the same number of p-values.
    parser.add_argument("--target_pvalues", type=int, default=None,
                        help="dynamic mode: keep adding trials until this "
                             "many successful trials (p-values) are "
                             "collected; ignores --iter / --seed_end and "
                             "consumes seeds sequentially from --seed_start")
    parser.add_argument("--max_trials", type=int, default=None,
                        help="cap on trials per condition in dynamic mode "
                             "(default: 50x target)")
    parser.add_argument("--out", type=str, default="result/realdata_results.csv")
    # Hyperparameters (same defaults as the synthetic experiments)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="lasso regularization strength in the "
                             "(1/(2n)) RSS + alpha*|beta|_1 convention")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--ms_k", type=int, default=3,
                        help="size of the set selected by marginal screening")
    # Lambda grid: logspace(lam_lo, lam_hi, n_lambda) = {0.01, 0.032, 0.1}
    parser.add_argument("--n_lambda", type=int, default=3)
    parser.add_argument("--lam_lo", type=float, default=-2.0)
    parser.add_argument("--lam_hi", type=float, default=-1.0)
    args = parser.parse_args()

    cfg = {
        "alpha": args.alpha, "max_iter": args.max_iter, "tol": args.tol,
        "K": args.K, "n_lambda": args.n_lambda,
        "lam_lo": args.lam_lo, "lam_hi": args.lam_hi,
        "ms_k": args.ms_k,
    }

    methods = [m for m in args.methods.split(",") if m]
    datasets = [d for d in args.datasets.split(",") if d]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method: {m} (choices: {METHODS})")
    for d in datasets:
        if d not in DATASET_NAMES:
            raise ValueError(f"unknown dataset: {d} (choices: {DATASET_NAMES})")

    seed_start = args.seed_start if args.seed_start is not None else 0
    seed_end = args.seed_end if args.seed_end is not None else args.iter

    records = []
    n_failed = 0
    if args.target_pvalues:
        # Dynamic mode: per (method x dataset), add trials until the number
        # of successful trials reaches the target.
        print(f"methods={methods} datasets={datasets} n={args.n_samples} "
              f"dynamic mode: target {args.target_pvalues} p-values "
              f"(seeds sequential from {seed_start}+{args.seed_offset})")
        max_trials = args.max_trials or 50 * args.target_pvalues
        for method in methods:
            for dataset in datasets:
                recs, failed = run_dynamic(
                    method, dataset, args.n_samples, cfg,
                    args.target_pvalues, seed_start, args.seed_offset,
                    max(args.workers, 1), max_trials,
                )
                records.extend(recs)
                n_failed += failed
        n_total = len(records) + n_failed
    else:
        tasks = [
            (method, dataset, args.n_samples, seed + args.seed_offset, cfg)
            for method in methods
            for dataset in datasets
            for seed in range(seed_start, seed_end)
        ]
        print(f"methods={methods} datasets={datasets} n={args.n_samples} "
              f"seeds=[{seed_start},{seed_end})+{args.seed_offset} "
              f"-> {len(tasks)} tasks")

        if args.workers <= 1:
            for task in tqdm(tasks, desc="realdata"):
                rec = run_single(task)
                if rec is not None:
                    records.append(rec)
                else:
                    n_failed += 1
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(run_single, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(tasks),
                                desc="realdata"):
                    # Even a crashed worker process must not stop the run
                    try:
                        rec = fut.result()
                    except Exception:
                        traceback.print_exc()
                        rec = None
                    if rec is not None:
                        records.append(rec)
                    else:
                        n_failed += 1
        n_total = len(tasks)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    failed_note = f" ({n_failed}/{n_total} failed trials skipped)" if n_failed else ""
    print(f"\nsaved {len(df)} rows -> {args.out}{failed_note}")

    # Rejection rates per dataset (the power proxy on real data)
    if not df.empty:
        summary = df.groupby(["method", "dataset"]).agg(
            n=("p_value", "size"),
            reject_proposed=("p_value", lambda s: float((s < 0.05).mean())),
            reject_oc=("oc_p_value", lambda s: float((s < 0.05).mean())),
            reject_bonf=("bonferroni_p_value", lambda s: float((s < 0.05).mean())),
            mean_time=("time", "mean"),
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
