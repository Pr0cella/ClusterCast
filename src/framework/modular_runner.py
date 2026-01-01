from __future__ import annotations

"""
Quick-start CLI to run the new modular framework with built-in model adapters.

Usage (from src directory):
  python -m framework.modular_runner --model hybrid --horizon 13 --opt-budget 50

This will:
  - Load/prepare data using forecast.py helpers
  - Register the modular adapters (baseline/contagion/hybrid)
  - Optionally optimize model params via random search on a small window
  - Run a small rolling forecast for the specified horizon
  - Print a concise summary (and optionally write predictions)
"""

import os
import argparse
import numpy as np
import pandas as pd

from .registry import register_model
from .adapters import BaselineAdapter, ContagionAdapter, HybridAdapter
from .simple_opt_eval import RandomSearchOptimizer, SimpleEvaluator
from .reporter import SplitReporter
from .base import Artifacts
import forecast as fc
from .plugin_loader import discover_and_register_models

# Import data helpers from forecast.py
from forecast import prepare_data, calculate_log_loss


def build_context(Y_t: pd.DataFrame, t_index: int, T_train: int):
    Y_train = Y_t.iloc[t_index - T_train: t_index]
    Y_current = Y_t.iloc[t_index].values
    return {
        'Y_train': Y_train,
        'Y_current': Y_current,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run modular model(s) with optional optimization and reporting.")
    parser.add_argument('--model', default='hybrid', help='Model id to run (built-in: baseline|contagion|hybrid, or plugin id)')
    parser.add_argument('--horizon', type=int, default=1, help='Forecast horizon (weeks) to evaluate/forecast')
    parser.add_argument('--train-window', type=int, default=26, help='Training window length (weeks)')
    parser.add_argument('--opt-budget', type=int, default=0, help='Random search trials. 0 disables optimization')
    parser.add_argument('--data', default=None, help='Optional incidents CSV path (else auto-resolve)')
    parser.add_argument('--emit-reports', action='store_true', help='Write split predictions/calibration reports using forecast generators')
    parser.add_argument('--plugins-dir', default=os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'plugins', 'models')), help='Directory to search for model plugins')
    parser.add_argument('--list-models', action='store_true', help='List available models (including plugins) and exit')
    args = parser.parse_args(argv)

    # Register adapters (so users could extend by importing their own and calling register_model)
    register_model(BaselineAdapter)
    register_model(ContagionAdapter)
    register_model(HybridAdapter)

    # Discover plugin models (if any)
    discovered = discover_and_register_models(args.plugins_dir)

    # Pick model
    model_map = {
        'baseline': BaselineAdapter,
        'contagion': ContagionAdapter,
        'hybrid': HybridAdapter,
    }
    # If user passed a plugin id, allow it; otherwise fall back to built-ins
    from . import registry as reg
    if args.list_models:
        print("Available models:")
        for mid in reg.list_models():
            print(" -", mid)
        return 0

    if args.model in model_map:
        ModelCls = model_map[args.model]
    else:
        # Try plugin registry by id
        try:
            ModelCls = reg.get_model(args.model)
        except Exception:
            print(f"[ERROR] Unknown model id '{args.model}'. Use --list-models to see options.")
            return 2
    model = ModelCls(horizon_weeks=args.horizon)

    # Prepare data
    # Need enough to cover train window + horizon
    required_history = max(args.train_window, 8)
    Y_t, n_targets, total_weeks, _ = prepare_data(args.data, required_history, args.horizon)
    if Y_t is None or total_weeks < args.train_window + args.horizon:
        print("[ERROR] Not enough data for the requested windows.")
        return 1

    # Optional: optimize via simple random search on a short objective (NLL on last horizon)
    if args.opt_budget > 0 and model.get_search_space():
        optimizer = RandomSearchOptimizer()
        # objective: average NLL over the last <horizon> steps
        def objective(params):
            model.set_params(**params)
            # simulate over the last horizon using a fixed training window
            H_state = model.fit(Y_t.iloc[total_weeks - args.train_window - args.horizon: total_weeks - args.horizon])
            nlls = []
            for step in range(args.horizon):
                t = total_weeks - args.horizon + step
                ctx = build_context(Y_t, t, args.train_window)
                p = model.predict_proba(H_state, ctx)
                y = (Y_t.iloc[t].values > 0).astype(int)
                nlls.append(calculate_log_loss(y, p))
            return float(np.mean(nlls))

        best, trials = optimizer.optimize(model, None, objective, budget=args.opt_budget)
        if best:
            model.set_params(**best)
        print(f"Optimization done. Best params: {best}")

    # Fit on last training window and run a forward forecast for <horizon>
    start = total_weeks - args.horizon
    H_state = model.fit(Y_t.iloc[start - args.train_window: start])
    preds = []
    for step in range(args.horizon):
        t = start + step
        ctx = build_context(Y_t, t, args.train_window)
        p = model.predict_proba(H_state, ctx)
        preds.append(p)

    preds = np.array(preds)
    avg_weekly = np.mean(preds, axis=0)

    # Quick summary
    print(f"Model: {model.id} | Horizon: {args.horizon} | Train: {args.train_window} weeks")
    print("Params:", model.get_params())
    s = pd.Series(avg_weekly, index=Y_t.columns)
    top = s.sort_values(ascending=False).head(10)
    print("Top targets (avg weekly prob):")
    for tgt, val in top.items():
        print(f" - {tgt}: {val:.2%}")

    # Optional: emit split reports using the same report generators
    if args.emit_reports:
        # Build a minimal forecast_results payload compatible with forecast.generate_risk_report
        forecast_results = [{
            'model_config': f'{model.id} (modular)',
            'forecast_date': pd.Timestamp.now(),
            'forecast_start_week': Y_t.index[total_weeks - 1] + pd.Timedelta(weeks=1),
            'forecast_horizon_weeks': args.horizon,
            'forecast_probabilities': np.array(preds),
            'average_weekly_probabilities': avg_weekly,
            'target_labels': list(Y_t.columns),
            'optimal_decay': model.get_params().get('decay'),
            'optimal_jump': model.get_params().get('jump'),
            # calibration fields omitted in predictions-only mode
            'calibration_applied': False,
            'calibration_method': None,
            'calibration_source': None,
            'calibration_param': None,
            'calibration_smoothing_alpha': None,
        }]

        rep = SplitReporter()
        artifacts = Artifacts(run_id=pd.Timestamp.now().strftime('%Y%m%d-%H%M%S'), output_dir='output')
        rep.render_predictions(artifacts, forecast_results=forecast_results, export_md=True, export_json=True)
        # Calibration report summarizes existing artifacts already built by forecast’s backtests
        rep.render_calibration(artifacts, export_md=True, export_json=True)
        print("Reports written:")
        print(" -", artifacts.predictions_md)
        print(" -", artifacts.predictions_json)
        print(" -", artifacts.calibration_md)
        print(" -", artifacts.calibration_json)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
