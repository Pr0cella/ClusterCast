from __future__ import annotations

"""
Rolling backtest CLI for modular models (built-ins + plugins).

Examples (from src):
    # Single model
    python -m framework.modular_backtest --model my_model --train-window 26 --test-window 13 --opt-budget 10

    # Multiple models
    python -m framework.modular_backtest --model my_model --model hybrid --train-window 26 --test-window 13 --export-csv --export-md

    # All discovered models
    python -m framework.modular_backtest --all-models --train-window 26 --test-window 13 --out-dir output

This will:
  - Load data via forecast.prepare_data
  - Optionally optimize model params with a small validation slice
  - Run a one-step-ahead rolling backtest over the last <test-window> weeks
  - Print ECE, Brier, NLL and Skill vs Baseline
"""

import argparse
import os
from typing import List
import numpy as np
import pandas as pd

from .registry import register_model
from .adapters import BaselineAdapter, ContagionAdapter, HybridAdapter
from .simple_opt_eval import RandomSearchOptimizer, SimpleEvaluator
from .plugin_loader import discover_and_register_models
from .base import ModelInterface

# Reuse helpers from forecast.py
from forecast import prepare_data, calculate_log_loss, baseline_forecast


def build_context(Y_t: pd.DataFrame, t_index: int, T_train: int):
    Y_train = Y_t.iloc[t_index - T_train: t_index]
    Y_current = Y_t.iloc[t_index].values
    return {'Y_train': Y_train, 'Y_current': Y_current}


def run_backtest(model: ModelInterface, Y_t: pd.DataFrame, train_window: int, test_window: int,
                 opt_budget: int = 0) -> dict:
    total_weeks = len(Y_t)
    evaluator = SimpleEvaluator()

    # Optional quick-and-dirty param search using a tiny validation slice right before the test window
    if opt_budget > 0 and model.get_search_space():
        optimizer = RandomSearchOptimizer()
        # choose up to 3 validation steps to keep it fast
        val_steps = int(min(3, max(1, test_window // 4)))
        val_end = total_weeks - test_window
        val_start = max(train_window + val_steps, val_end - val_steps)
        if val_end - val_start >= 1:
            def objective(params):
                model.set_params(**params)
                nlls: List[float] = []
                for t in range(val_start, val_end):
                    if t - train_window < 0:
                        continue
                    state = model.fit(Y_t.iloc[t - train_window: t])
                    ctx = build_context(Y_t, t, train_window)
                    p = model.predict_proba(state, ctx)
                    y = (Y_t.iloc[t].values > 0).astype(int)
                    nlls.append(calculate_log_loss(y, p))
                return float(np.mean(nlls)) if nlls else float('inf')

            best, _ = optimizer.optimize(model, None, objective, budget=opt_budget)
            if best:
                model.set_params(**best)

    # Rolling one-step-ahead backtest over the last test_window weeks
    start = total_weeks - test_window
    preds = []
    labels = []
    preds_baseline = []
    for t in range(start, total_weeks):
        if t - train_window < 0:
            continue
        train_df = Y_t.iloc[t - train_window: t]
        state = model.fit(train_df)
        ctx = build_context(Y_t, t, train_window)
        p = model.predict_proba(state, ctx)
        y = (Y_t.iloc[t].values > 0).astype(int)
        preds.append(p)
        labels.append(y)
        # Baseline comparison from same train window
        p_base = baseline_forecast(train_df).values
        preds_baseline.append(p_base)

    preds = np.asarray(preds)
    labels = np.asarray(labels)
    preds_baseline = np.asarray(preds_baseline)

    # Aggregate metrics
    m = evaluator.evaluate(preds, labels)
    # Baseline NLL
    eps = 1e-15
    p_base = np.clip(preds_baseline.ravel(), eps, 1 - eps)
    y = labels.ravel()
    nll_base = float(-np.mean(y * np.log(p_base) + (1 - y) * np.log(1 - p_base)))
    skill = None
    if m.nll is not None and np.isfinite(nll_base) and nll_base > 0:
        skill = 100.0 * (nll_base - m.nll) / nll_base

    return {
        'ece': m.ece,
        'brier': m.brier,
        'nll': m.nll,
        'nll_baseline': nll_base,
        'skill_vs_baseline_percent': skill,
        'steps': int(len(preds)),
    }


def _export_results(rows: List[dict], out_dir: str, train_window: int, test_window: int,
                    export_csv: bool, export_md: bool) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    paths: List[str] = []
    df = pd.DataFrame(rows)
    df = df[[
        'model_id', 'train_window', 'test_window', 'steps',
        'ece', 'brier', 'nll', 'nll_baseline', 'skill_vs_baseline_percent'
    ]]
    base = f"backtest_results_tw{train_window}_hw{test_window}_{ts}"
    if export_csv:
        csv_path = os.path.join(out_dir, base + '.csv')
        df.to_csv(csv_path, index=False)
        paths.append(csv_path)
    if export_md:
        md_path = os.path.join(out_dir, base + '.md')
        # Simple markdown table
        md = "| " + " | ".join(df.columns) + " |\n"
        md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
        for _, r in df.iterrows():
            vals = [
                r['model_id'],
                str(r['train_window']),
                str(r['test_window']),
                str(r['steps']),
                f"{r['ece']:.6f}" if pd.notna(r['ece']) else "",
                f"{r['brier']:.6f}" if pd.notna(r['brier']) else "",
                f"{r['nll']:.6f}" if pd.notna(r['nll']) else "",
                f"{r['nll_baseline']:.6f}" if pd.notna(r['nll_baseline']) else "",
                f"{r['skill_vs_baseline_percent']:.2f}%" if pd.notna(r['skill_vs_baseline_percent']) else "",
            ]
            md += "| " + " | ".join(vals) + " |\n"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md)
        paths.append(md_path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rolling backtest for modular models (built-ins + plugins)")
    parser.add_argument('--model', action='append', help='Model id; repeat to test multiple models')
    parser.add_argument('--all-models', action='store_true', help='Run backtests for all discovered models')
    parser.add_argument('--train-window', type=int, default=26, help='Training window length (weeks)')
    parser.add_argument('--test-window', type=int, default=13, help='Test window length (weeks)')
    parser.add_argument('--opt-budget', type=int, default=0, help='Random search budget for a tiny pre-test validation')
    parser.add_argument('--data', default=None, help='Optional incidents CSV path (else auto-resolve)')
    parser.add_argument('--plugins-dir', default=os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'plugins', 'models')), help='Directory to search for model plugins')
    parser.add_argument('--out-dir', default='output', help='Directory to write exported results (CSV/MD)')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--export-md', action='store_true', help='Export results to Markdown table')
    args = parser.parse_args(argv)

    # Register built-ins
    register_model(BaselineAdapter)
    register_model(ContagionAdapter)
    register_model(HybridAdapter)

    # Discover plugins
    discover_and_register_models(args.plugins_dir)

    # Resolve model list
    from . import registry as reg
    model_ids: List[str] = []
    if args.all_models:
        model_ids = reg.list_models()
    elif args.model:
        model_ids = list(dict.fromkeys(args.model))  # dedupe, preserve order
    else:
        print("[ERROR] Provide at least one --model or use --all-models.")
        return 2

    # Prepare data: need enough history to cover train + test
    required_history = args.train_window + args.test_window
    Y_t, n_targets, total_weeks, _ = prepare_data(args.data, required_history, 1)
    if Y_t is None or total_weeks < required_history:
        print("[ERROR] Not enough data for the requested windows.")
        return 1

    rows: List[dict] = []
    for mid in model_ids:
        try:
            ModelCls = reg.get_model(mid)
        except Exception:
            print(f"[WARNING] Skipping unknown model id '{mid}'.")
            continue
        model = ModelCls(horizon_weeks=1)
        res = run_backtest(model, Y_t, args.train_window, args.test_window, opt_budget=args.opt_budget)

        print("\n--- Rolling Backtest Results ---")
        print(f"Model: {model.id}")
        print(f"Train window: {args.train_window} weeks | Test window: {args.test_window} weeks | Steps: {res['steps']}")
        print(f"Params: {model.get_params()}")
        print(f"ECE:   {res['ece']:.6f}" if res['ece'] is not None else "ECE:   n/a")
        print(f"Brier: {res['brier']:.6f}" if res['brier'] is not None else "Brier: n/a")
        print(f"NLL:   {res['nll']:.6f}" if res['nll'] is not None else "NLL:   n/a")
        print(f"NLL (baseline): {res['nll_baseline']:.6f}")
        if res['skill_vs_baseline_percent'] is not None:
            print(f"Skill vs Baseline: {res['skill_vs_baseline_percent']:.2f}%")
        else:
            print("Skill vs Baseline: n/a")

        rows.append({
            'model_id': model.id,
            'train_window': args.train_window,
            'test_window': args.test_window,
            **res,
        })

    if rows and (args.export_csv or args.export_md):
        out_paths = _export_results(rows, args.out_dir, args.train_window, args.test_window, args.export_csv, args.export_md)
        print("\nExports written:")
        for p in out_paths:
            print(" -", p)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
