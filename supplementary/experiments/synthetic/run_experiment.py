"""Synthetic-data experiments: run the three feature-selection methods
(marginal_screening / lasso / cv_r2_lasso) on iid or correlated data and
write the p-values and computation times to a CSV file.

Each trial performs
  1. data generation (null when signal=0)
  2. one run of the algorithm on the observed data to obtain eta
  3. p-value computation via si.inference (this interval is timed)
and appends one row of results.

Workers are top-level functions (Windows multiprocessing compatible) and
regenerate the data inside each worker instead of pickling tracked arrays.

Two execution modes:
  fixed:    --iter (or --seed_start/--seed_end) fixes the number of trials
  dynamic:  --target_pvalues N keeps adding trials until N valid p-values
            (trials whose selection succeeded) are collected (for power)

Usage:
  python experiments/synthetic/run_experiment.py --iter 200 --workers 1
  python experiments/synthetic/run_experiment.py --signal 0.2 --target_pvalues 110
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import before numpy/BLAS so that every process is pinned to one thread
# (set_env runs at import time).
from experiments.common.environment import set_env, set_seed

import time
import argparse
import traceback
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

import autosi as si
from experiments.synthetic.data import generate_data
from experiments.common import methods as methods_lib
from experiments.common.methods import marginal_screening, lasso, cv_r2_lasso

METHODS = ["marginal_screening", "lasso", "cv_r2_lasso"]
DATA_TYPES = ["iid", "corr"]


def _build_algorithm(method, X, cfg):
    """Build an algorithm function of y alone for the given method name.

    pick_u is the random number deciding which member of the selected set is
    tested; it is fixed from the seed independently of y (injected into cfg
    by run_single).
    """
    n, p = X.shape
    if method == "marginal_screening":
        return partial(marginal_screening, X=X, k=cfg["ms_k"],
                       pick_u=cfg["pick_u"])
    if method == "lasso":
        return partial(
            lasso, X=X, n_samples=n, n_features=p,
            alpha=cfg["alpha"], max_iter=cfg["max_iter"], tol=cfg["tol"],
            pick_u=cfg["pick_u"],
        )
    if method == "cv_r2_lasso":
        # The folds are fixed per trial so that the CV partition is the same
        # when computing eta and during inference.
        rng = np.random.default_rng(cfg["fold_seed"])
        fold_indices = np.array_split(rng.permutation(n), cfg["K"])
        lambda_grid = np.logspace(cfg["lam_lo"], cfg["lam_hi"], cfg["n_lambda"])
        return partial(
            cv_r2_lasso, X=X, n_samples=n, n_features=p,
            fold_indices=fold_indices, lambda_grid=lambda_grid,
            K=cfg["K"], max_iter=cfg["max_iter"], tol=cfg["tol"],
            pick_u=cfg["pick_u"],
        )
    raise ValueError(f"unknown method: {method}")


def _data_split_p_value(method, X_np, y_np, Sigma, cfg):
    """Data splitting baseline: select on one half, z-test on the other.

    Uses the same n rows as the other methods, half for selection and half
    for inference, so the total data usage is fair. The split interleaves
    the observations (odd/even positions). Under iid the rows are
    exchangeable and the classical z-test is valid; under corr (AR(1)) the
    two halves are dependent (adjacent observations correlate the most), so
    data splitting fails to control the type I error, illustrating its
    limitation on dependent data.

    Returns
    -------
    (p_value, selected_index)
    """
    n = X_np.shape[0]
    idx1 = np.arange(0, n, 2)
    idx2 = np.arange(1, n, 2)
    X1_np, X2_np = X_np[idx1], X_np[idx2]
    y1_np, y2_np = y_np[idx1], y_np[idx2]

    # First half: run the same selection algorithm once
    X1 = si.siarray(X1_np)
    algorithm = _build_algorithm(method, X1, cfg)
    methods_lib.last_active_set = None
    eta1 = algorithm(si.siarray(y1_np))
    ds_active = methods_lib.last_active_set
    if ds_active is not None and len(ds_active) == 0:
        raise RuntimeError("data splitting: no feature selected on first half")
    selected = int(methods_lib.last_picked)

    # Second half: z-test of the selected-model regression coefficient
    # (same estimand as the proposed method).
    pos = ds_active.index(selected)
    X2_M = X2_np[:, ds_active]
    eta2 = np.linalg.solve(X2_M.T @ X2_M, X2_M.T)[pos]
    stat = float(eta2 @ y2_np)
    if np.isscalar(Sigma):
        var_stat = float(Sigma) * float(eta2 @ eta2)
    else:
        # Submatrix of the covariance for the second (even-position) half
        var_stat = float(eta2 @ Sigma[np.ix_(idx2, idx2)] @ eta2)
    p_value = float(2.0 * norm.sf(abs(stat) / np.sqrt(var_stat)))
    return p_value, selected


def run_single(task):
    """One trial: generate data -> obtain eta -> compute p-values (timed)."""
    method, data_type, seed, cfg = task
    set_env(parallel=True)
    set_seed(seed)
    try:
        X_np, y_np, Sigma = generate_data(
            data_type=data_type,
            n_samples=cfg["n_samples"],
            n_features=cfg["n_features"],
            seed=seed,
            signal=cfg["signal"],
            n_signal=cfg["n_signal"],
        )
        X = si.siarray(X_np)
        y = si.siarray(y_np)

        # Random number for drawing the tested feature; separated from the
        # data-generation seed and the CV folds via a seed sequence.
        pick_u = float(np.random.default_rng([seed, 987654321]).uniform())
        cfg_local = dict(cfg, fold_seed=seed, pick_u=pick_u)
        algorithm = _build_algorithm(method, X, cfg_local)

        # Active set on the observed data (lasso variants only). Reset it so
        # that no value from a previous task leaks, and read it before the
        # data-splitting run or the inference overwrites it.
        methods_lib.last_active_set = None
        eta = algorithm(y)
        active_set = methods_lib.last_active_set
        # A trial with an empty selection has nothing to test; skip it.
        if active_set is not None and len(active_set) == 0:
            raise RuntimeError("empty selected feature set")

        # Feature drawn for testing on the observed data
        selected_index = int(methods_lib.last_picked)
        # The truly active features are the first n_signal ones (only when
        # signal != 0; the null has none). Rejections with is_signal=False
        # are type I errors and are excluded from the power aggregation.
        n_active = min(cfg["n_signal"], cfg["n_features"]) if cfg["signal"] != 0.0 else 0
        is_signal = bool(selected_index < n_active)

        # Data splitting baseline (cheap: one selection + one z-test).
        # Selecting on half the data, its selection can differ from the
        # proposed method's, hence the separate ds_is_signal flag.
        ds_p, ds_selected = _data_split_p_value(
            method, X_np, y_np, Sigma, cfg_local
        )
        ds_is_signal = bool(ds_selected < n_active)

        # In the power experiments (signal != 0), trials where a null
        # feature was drawn are excluded from the conditional power anyway,
        # so the costly SI inference is skipped (empty cells in the CSV).
        # naive / bonferroni / oc are aggregated under the same is_signal
        # filter and can be skipped together. skip_si marks trials launched
        # only to top up the data-splitting count in dynamic mode.
        if cfg["signal"] != 0.0 and (not is_signal or cfg.get("skip_si", False)):
            p_value = oc_p = naive_p = bonferroni_p = elapsed = None
        else:
            oc_p = float(si.inference(eta=eta, prob_vec=y, var=Sigma, algorithm=algorithm, inference_mode="over_conditioning").p_value)
            start = time.time()
            result = si.inference(eta=eta, prob_vec=y, var=Sigma, algorithm=algorithm)
            elapsed = time.time() - start

            p_value = float(result.p_value)
            naive_p = float(result.naive_p_value())
            # bonferroni_p_value takes the log of the number of comparisons;
            # the number of possible selections is bounded by 2^p, so pass
            # log(2^p) = p * log(2).
            bonferroni_p = float(result.bonferroni_p_value(
                log_num_comparisons=cfg["n_features"] * np.log(2)
            ))

        return {
            "method": method,
            "data_type": data_type,
            "seed": seed,
            "p_value": p_value,
            "oc_p_value": oc_p,
            "naive_p_value": naive_p,
            "bonferroni_p_value": bonferroni_p,
            "time": elapsed,
            "n_samples": cfg["n_samples"],
            "n_features": cfg["n_features"],
            "signal": cfg["signal"],
            "n_signal": cfg["n_signal"],
            # Recorded so each CSV is self-describing about its settings
            "K": cfg["K"],
            "n_lambda": cfg["n_lambda"],
            "lam_lo": cfg["lam_lo"],
            "lam_hi": cfg["lam_hi"],
            # Needed to compute power over trials that picked a true signal
            "selected_index": selected_index,
            "is_signal": is_signal,
            # Active set ("|"-separated; empty for marginal screening)
            "active_set": "" if active_set is None else "|".join(map(str, active_set)),
            "active_size": None if active_set is None else len(active_set),
            # Data splitting results (its selection uses only the first half)
            "ds_p_value": ds_p,
            "ds_selected_index": ds_selected,
            "ds_is_signal": ds_is_signal,
        }
    except Exception:
        # A single failed trial (e.g., numerical issues) does not stop the
        # whole run; log it and continue.
        print(f"[skip] method={method} data_type={data_type} seed={seed} "
              "failed (continuing)", file=sys.stderr)
        traceback.print_exc()
        return None


def run_dynamic(method, data_type, cfg, target, seed_start, seed_offset,
                workers, max_trials):
    """Keep adding trials until `target` valid p-values are collected.

    In the power experiments only trials whose selection picked a signal
    feature count, so the numbers of valid p-values for the proposed family
    (is_signal=True) and for data splitting (ds_is_signal=True) are tracked
    separately until both reach the target (under the null every trial
    counts). Trials are submitted in waves sized from the observed selection
    success rate so that many workers stay busy. Data splitting selects on
    half the data and succeeds less often (the bottleneck), so the proposed
    side usually fills up first; later waves set skip_si and compute only
    the selection and the z-test, skipping the costly SI inference. Seeds
    are consumed sequentially from seed_start (shift it per job when
    splitting work across machines).
    """
    records = []
    n_failed = 0
    n_prop = 0   # count with is_signal=True (proposed/oc/naive/bonferroni)
    n_ds = 0     # count with ds_is_signal=True
    n_launched = 0
    next_seed = seed_start + seed_offset
    is_null = cfg["signal"] == 0.0

    cfg_skip = dict(cfg, skip_si=True)  # DS-only tasks (skip SI inference)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        wave = 0
        while True:
            # Size the wave from the remaining need divided by a smoothed
            # (optimistic) success rate, estimated separately for the
            # proposed side and the DS side; trials beyond the proposed need
            # exist only to top up DS and run with skip_si.
            n_done = len(records) + n_failed
            if is_null:
                need_full = need_total = target - len(records)
            else:
                r_prop = (n_prop + 1) / (n_done + 1)
                r_ds = (n_ds + 1) / (n_done + 1)
                need_full = (target - n_prop) / r_prop   # trials incl. SI
                need_total = max(need_full, (target - n_ds) / r_ds)
            if need_total <= 0:
                break
            n_wave = min(int(need_total * 1.1) + 1, max_trials - n_launched)
            if n_wave <= 0:
                print(f"warning: reached max_trials={max_trials}; "
                      f"short of valid p-values "
                      f"(proposed={n_prop}, ds={n_ds} / target {target})",
                      file=sys.stderr)
                break
            wave += 1
            # The first n_full trials of the wave include SI; the rest are
            # DS-only.
            if need_full > 0:
                n_full = min(n_wave, int(need_full * 1.1) + 1)
            else:
                n_full = 0
            if n_wave > n_full:
                print(f"{method}/{data_type} wave{wave}: {n_full} with SI + "
                      f"{n_wave - n_full} DS-only trials "
                      f"(valid p-values proposed={n_prop}, ds={n_ds} / "
                      f"target {target})")
            tasks = [(method, data_type, s, cfg if j < n_full else cfg_skip)
                     for j, s in enumerate(range(next_seed, next_seed + n_wave))]
            next_seed += n_wave
            n_launched += n_wave
            futures = [ex.submit(run_single, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"{method}/{data_type} wave{wave}"):
                try:
                    rec = fut.result()
                except Exception:
                    traceback.print_exc()
                    rec = None
                if rec is None:
                    n_failed += 1
                    continue
                records.append(rec)
                # The proposed side counts p-values actually computed
                # (skip_si waves add none even when is_signal=True)
                n_prop += int(rec["p_value"] is not None)
                n_ds += int(rec["ds_is_signal"])

    print(f"{method}/{data_type}: {n_launched} trials for "
          f"proposed={n_prop}, ds={n_ds} valid p-values (target {target})")
    return records, n_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=200,
                        help="number of trials per condition")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of parallel workers")
    # Job splitting (run a subset of conditions/seeds per machine)
    parser.add_argument("--methods", type=str, default=",".join(METHODS),
                        help="comma-separated method names (default: all)")
    parser.add_argument("--data_types", type=str, default=",".join(DATA_TYPES),
                        help="comma-separated data types (default: all)")
    parser.add_argument("--seed_start", type=int, default=None,
                        help="start of the seed range (default: 0)")
    parser.add_argument("--seed_end", type=int, default=None,
                        help="end of the seed range (exclusive; default: --iter)")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="constant added to every seed; vary it per "
                             "method to keep the datasets independent")
    # Dynamic mode (for power): only trials with a successful selection
    # contribute p-values, so the target is a number of valid p-values.
    parser.add_argument("--target_pvalues", type=int, default=None,
                        help="dynamic mode: keep adding trials until this "
                             "many valid p-values (is_signal / ds_is_signal "
                             "separately) are collected; ignores --iter / "
                             "--seed_end and consumes seeds sequentially "
                             "from --seed_start")
    parser.add_argument("--max_trials", type=int, default=None,
                        help="cap on trials in dynamic mode (default: 50x "
                             "target); guards against settings with a very "
                             "low selection success rate")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_features", type=int, default=20,
                        help="number of features p")
    parser.add_argument("--signal", type=float, default=0.0,
                        help="true coefficient magnitude Delta (0 = null)")
    parser.add_argument("--n_signal", type=int, default=3,
                        help="number of leading features carrying the signal")
    parser.add_argument("--ms_k", type=int, default=3,
                        help="size of the set selected by marginal screening")
    parser.add_argument("--out", type=str, default="result/experiment_results.csv")
    # Hyperparameters shared by the lasso variants
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="lasso regularization strength in the "
                             "(1/(2n)) RSS + alpha*|beta|_1 convention")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    # cv_r2_lasso-specific: the lambda grid is
    # logspace(lam_lo, lam_hi, n_lambda) = {0.01, 0.032, 0.1}. All values
    # were fixed a priori and not tuned.
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_lambda", type=int, default=3)
    parser.add_argument("--lam_lo", type=float, default=-2.0)
    parser.add_argument("--lam_hi", type=float, default=-1.0)
    args = parser.parse_args()

    cfg = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "signal": args.signal,
        "n_signal": args.n_signal,
        "ms_k": args.ms_k,
        "alpha": args.alpha,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "K": args.K,
        "n_lambda": args.n_lambda,
        "lam_lo": args.lam_lo,
        "lam_hi": args.lam_hi,
    }

    methods = [m for m in args.methods.split(",") if m]
    data_types = [d for d in args.data_types.split(",") if d]
    seed_start = args.seed_start if args.seed_start is not None else 0
    seed_end = args.seed_end if args.seed_end is not None else args.iter

    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method: {m} (choices: {METHODS})")
    for d in data_types:
        if d not in DATA_TYPES:
            raise ValueError(f"unknown data_type: {d} (choices: {DATA_TYPES})")

    records = []
    n_failed = 0
    if args.target_pvalues:
        # Dynamic mode: per (method x data_type), add trials until the
        # number of valid p-values reaches the target.
        print(f"methods={methods} data_types={data_types} "
              f"dynamic mode: target {args.target_pvalues} p-values "
              f"(seeds sequential from {seed_start}+{args.seed_offset})")
        max_trials = args.max_trials or 50 * args.target_pvalues
        for method in methods:
            for data_type in data_types:
                recs, failed = run_dynamic(
                    method, data_type, cfg, args.target_pvalues,
                    seed_start, args.seed_offset,
                    max(args.workers, 1), max_trials,
                )
                records.extend(recs)
                n_failed += failed
        n_total = len(records) + n_failed
    else:
        # Fixed mode: enumerate (method x data_type x seed) tasks.
        tasks = [
            (method, data_type, seed + args.seed_offset, cfg)
            for method in methods
            for data_type in data_types
            for seed in range(seed_start, seed_end)
        ]
        print(f"methods={methods} data_types={data_types} "
              f"seeds=[{seed_start},{seed_end})+{args.seed_offset} "
              f"-> {len(tasks)} tasks")

        if args.workers <= 1:
            for task in tqdm(tasks, desc="experiments"):
                rec = run_single(task)
                if rec is not None:
                    records.append(rec)
                else:
                    n_failed += 1
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(run_single, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(tasks),
                                desc="experiments"):
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

    # Quick per-condition summary
    if not df.empty:
        summary = df.groupby(["method", "data_type"]).agg(
            n=("p_value", "size"),
            mean_p=("p_value", "mean"),
            # Rows with skipped SI inference have empty p-values; drop them
            fpr_005=("p_value", lambda s: float((s.dropna() < 0.05).mean())),
            ds_005=("ds_p_value", lambda s: float((s < 0.05).mean())),
            # Fraction of trials that drew a signal feature (0 under null)
            signal_rate=("is_signal", "mean"),
            ds_signal_rate=("ds_is_signal", "mean"),
            mean_time=("time", "mean"),
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
