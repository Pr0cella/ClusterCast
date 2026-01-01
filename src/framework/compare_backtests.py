from __future__ import annotations

"""
CLI to compare two modular backtest CSV exports and summarize improvements.

Usage (from src):
    # Explicit files
    python -m framework.compare_backtests --old output/backtest_results_old.csv --new output/backtest_results_new.csv --export-csv --export-md --out-dir output

    # Glob patterns (latest match picked for each)
    python -m framework.compare_backtests --old "output/backtest_results_tw26_hw13_*.csv" --new "output/backtest_results_tw26_hw13_*.csv" --export-md

Outputs:
    - Per-model deltas for ECE, Brier, NLL, and Skill vs Baseline
    - Aggregate summary (mean/median deltas)
"""

import argparse
import glob
import os
import pandas as pd
import numpy as np
from typing import List


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column casing if needed
    return df


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'model_id' in df.columns:
        df['model_id'] = df['model_id'].astype(str).str.strip().str.lower()
    if 'train_window' in df.columns:
        df['train_window'] = pd.to_numeric(df['train_window'], errors='coerce').astype('Int64')
    if 'test_window' in df.columns:
        df['test_window'] = pd.to_numeric(df['test_window'], errors='coerce').astype('Int64')
    return df
def _resolve_csv(path: str) -> str:
    # If it's a directory, pick latest backtest_results_*.csv inside immediately
    if os.path.isdir(path):
        inside = glob.glob(os.path.join(path, 'backtest_results_*.csv'))
        inside = [p for p in inside if os.path.isfile(p)]
        if inside:
            inside.sort(key=lambda p: os.path.getmtime(p))
            return inside[-1]
    # Direct file
    if os.path.isfile(path):
        return path
    # Glob patterns (filter out directories defensively)
    matches = [p for p in glob.glob(path) if os.path.isfile(p)]
    if matches:
        matches.sort(key=lambda p: os.path.getmtime(p))
        return matches[-1]
    # Help user by listing available files
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    out_dir = os.path.normpath(out_dir)
    available = []
    if os.path.isdir(out_dir):
        available = sorted(glob.glob(os.path.join(out_dir, 'backtest_results_*.csv')))
    msg = [f"[ERROR] Could not resolve CSV: {path}"]
    if available:
        msg.append("Available exports under ./output (examples):")
        for p in available[-10:]:
            msg.append(f" - {p}")
        msg.append("Tip: you can pass a glob like 'output/backtest_results_tw26_hw13_*.csv'")
    raise FileNotFoundError("\n".join(msg))



def _merge_on_model(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    keys = ['model_id', 'train_window', 'test_window']
    for k in keys:
        if k not in old.columns or k not in new.columns:
            raise ValueError(f"Missing key column '{k}' in input CSVs")
    merged = pd.merge(old, new, on=keys, how='inner', suffixes=('_old', '_new'))
    return merged


def _compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    # Compute deltas (new - old) for metrics
    out = pd.DataFrame()
    out['model_id'] = df['model_id']
    out['train_window'] = df['train_window']
    out['test_window'] = df['test_window']
    for col in ['ece', 'brier', 'nll', 'nll_baseline', 'skill_vs_baseline_percent']:
        if f'{col}_old' in df.columns and f'{col}_new' in df.columns:
            out[f'delta_{col}'] = df[f'{col}_new'] - df[f'{col}_old']
    return out


def _aggregate(deltas: pd.DataFrame) -> pd.DataFrame:
    # Mean and median for each delta column
    agg_rows = []
    for stat_name, fn in [('mean', np.nanmean), ('median', np.nanmedian)]:
        row = {'stat': stat_name}
        for col in deltas.columns:
            if col.startswith('delta_'):
                row[col] = float(fn(deltas[col])) if len(deltas[col]) > 0 else np.nan
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def _export(deltas: pd.DataFrame, summary: pd.DataFrame, out_dir: str, export_csv: bool, export_md: bool) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    paths: List[str] = []
    base = f'backtest_compare_{ts}'
    if export_csv:
        p1 = os.path.join(out_dir, base + '_deltas.csv')
        p2 = os.path.join(out_dir, base + '_summary.csv')
        deltas.to_csv(p1, index=False)
        summary.to_csv(p2, index=False)
        paths += [p1, p2]
    if export_md:
        def to_md(df: pd.DataFrame) -> str:
            md = "| " + " | ".join(df.columns) + " |\n"
            md += "| " + " | ".join(["---"] * len(df.columns)) + " |\n"
            for _, r in df.iterrows():
                vals = [str(r[c]) if not isinstance(r[c], float) else (f"{r[c]:.6f}" if not np.isnan(r[c]) else '') for c in df.columns]
                md += "| " + " | ".join(vals) + " |\n"
            return md
        p1 = os.path.join(out_dir, base + '_deltas.md')
        p2 = os.path.join(out_dir, base + '_summary.md')
        with open(p1, 'w', encoding='utf-8') as f:
            f.write(to_md(deltas))
        with open(p2, 'w', encoding='utf-8') as f:
            f.write(to_md(summary))
        paths += [p1, p2]
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Compare two backtest result CSVs and summarize deltas')
    parser.add_argument('--old', required=True, help='Path to the older backtest CSV')
    parser.add_argument('--new', required=True, help='Path to the newer backtest CSV')
    parser.add_argument('--out-dir', default='output', help='Directory to write comparison outputs')
    parser.add_argument('--export-csv', action='store_true', help='Export deltas and summary to CSV')
    parser.add_argument('--export-md', action='store_true', help='Export deltas and summary to Markdown')
    parser.add_argument('--tolerance', type=float, default=0.0, help='Only consider changes with absolute delta >= tolerance')
    parser.add_argument('--only-changed', action='store_true', help='Print only rows with any metric change >= tolerance')
    parser.add_argument('--model', action='append', help='Restrict comparison to specific model id(s); repeat or comma-separate')
    args = parser.parse_args(argv)

    old_path = _resolve_csv(args.old)
    new_path = _resolve_csv(args.new)

    # If both specs resolve to the same file, try to pick the previous CSV for 'old'
    if os.path.samefile(old_path, new_path):
        # Build a candidate list from the most specific source we can infer
        candidates = []
        # If user provided a directory or glob, use that; otherwise use the directory of the resolved file
        def list_candidates(spec: str, resolved: str) -> list[str]:
            if os.path.isdir(spec):
                pats = os.path.join(spec, 'backtest_results_*.csv')
                items = [p for p in glob.glob(pats) if os.path.isfile(p)]
                return items
            g = glob.glob(spec)
            if g:
                return [p for p in g if os.path.isfile(p)]
            # Fallback to the directory of the resolved path
            base_dir = os.path.dirname(resolved)
            items = [p for p in glob.glob(os.path.join(base_dir, 'backtest_results_*.csv')) if os.path.isfile(p)]
            return items

        candidates = list_candidates(args.old, old_path)
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p))
        if len(candidates) >= 2:
            old_path, new_path = candidates[-2], candidates[-1]
        else:
            raise FileNotFoundError("[ERROR] Only one matching CSV found; cannot compute deltas. Export at least two runs or pass distinct files.")

    print(f"Comparing files:\n - OLD: {old_path}\n - NEW: {new_path}")
    old = _normalize_keys(_read_csv(old_path))
    new = _normalize_keys(_read_csv(new_path))

    # Optional model filter
    selected_models = []
    if args.model:
        for m in args.model:
            selected_models += [s.strip() for s in str(m).split(',') if s.strip()]
        selected_models = sorted(set(selected_models))
        if 'model_id' in old.columns:
            old = old[old['model_id'].isin([m.lower() for m in selected_models])]
        if 'model_id' in new.columns:
            new = new[new['model_id'].isin([m.lower() for m in selected_models])]

    merged = _merge_on_model(old, new)
    if merged.empty:
        # No overlap; help the user
        def windows(df: pd.DataFrame):
            if {'train_window','test_window'}.issubset(df.columns):
                return sorted({(int(tw), int(hw)) for tw, hw in zip(df['train_window'], df['test_window'])})
            return []
        def models(df: pd.DataFrame):
            return sorted(set(df['model_id'])) if 'model_id' in df.columns else []
        print('[WARNING] No overlapping (model_id, train_window, test_window) rows between the two files.')
        print('Old file windows:', windows(old))
        print('New file windows:', windows(new))
        print('Old file models:', models(old))
        print('New file models:', models(new))
        print("Tip: compare runs with the same train/test windows (e.g., tw26/hw13), or use a glob like 'output/backtest_results_tw26_hw13_*.csv'.")

    deltas = _compute_deltas(merged)
    # Apply tolerance filter if requested
    if args.tolerance > 0 or args.only_changed:
        def row_changed(r: pd.Series) -> bool:
            for c in r.index:
                if c.startswith('delta_') and pd.notna(r[c]) and abs(float(r[c])) >= args.tolerance:
                    return True
            return False
        changed_mask = deltas.apply(row_changed, axis=1)
        filtered = deltas[changed_mask].reset_index(drop=True)
        if args.only_changed:
            deltas_to_print = filtered
        else:
            deltas_to_print = deltas
    else:
        deltas_to_print = deltas
    summary = _aggregate(deltas)

    print('\nPer-model metric deltas (new - old):')
    if args.only_changed and deltas_to_print.empty and not deltas.empty:
        print('[INFO] No changes exceed the given tolerance.')
    else:
        print(deltas_to_print.to_string(index=False))
    print('\nAggregate summary (mean/median deltas):')
    print(summary.to_string(index=False))

    if args.export_csv or args.export_md:
        paths = _export(deltas, summary, args.out_dir, args.export_csv, args.export_md)
        print('\nExports written:')
        for p in paths:
            print(' -', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
