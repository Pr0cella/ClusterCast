# -*- coding: utf-8 -*-
"""
### 1. File Structure

├── forecast.py                         # Main forecasting engine
├── self_optimizing_backtest.py         # Standalone backtesting module
├── framework/                          # Modular framework (interfaces, adapters, optimizer, reporter, validators)
│   ├── base.py                         # Model/Calibrator/Optimizer/Evaluator/Reporter interfaces and dataclasses
│   ├── adapters.py                     # Built-in adapters: baseline, contagion, hybrid
│   ├── registry.py                     # Model and calibrator registry
│   ├── simple_opt_eval.py              # RandomSearch optimizer + SimpleEvaluator (ECE/Brier/NLL)
│   ├── reporter.py                     # SplitReporter that writes predictions/calibration using forecast generators
│   ├── plugin_loader.py                # Plugin discovery: loads models from src/plugins/models/
│   ├── modular_runner.py               # CLI to run modular models (optimize, forecast, emit reports)
│   ├── modular_backtest.py             # CLI to run rolling backtests for any modular model
│   ├── compare_backtests.py            # CLI to compare two backtest result CSVs and summarize deltas
│   ├── scaffold_plugin.py              # CLI to scaffold a new plugin model file
│   └── validators.py                   # Output structure and link validators
├── plugins/
│   └── models/                         # Third-party model plugins (auto-discovered)
│       └── example_custom.py           # Example plugin model (id: custom_example)
├── data/
│   └── incidents.csv                   # Historical incident data
└── output/
  ├── predictions.md                    # Human-readable predictions report (presentation-focused)
  ├── predictions.json                  # Machine-readable predictions results
  ├── calibration.md                    # Human-readable calibration report (methods/diagnostics)
  ├── calibration.json                  # Machine-readable calibration summary and metadata
  └── calibration_assets/               # Calibration artifacts (used by calibration report and runtime)
      ├── calibration_summary.json        # ECE, Brier, NLL per model/horizon
      ├── calibration_method_comparison.json # Per-model method recommendation (+ params)
      ├── bins_*.csv                      # Bin stats incl. mean_pred, event_rate, bin_count, CI columns
      ├── reliability_*.png               # Reliability diagrams (raw + calibrated overlay, Wilson 95% CIs, bin n)
      ├── histogram_*.png                 # Probability histograms (raw + calibrated)
      └── segment_breakdown.csv           # Per-segment (Bundesland/Sector) ECE & Brier

### 2. Data Preparation

This section contains the function to load and prepare the incident data for the models.
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import sys
import subprocess
import re
import shutil

def find_data_file(preferred: str | None = None):
    """
    Best-effort resolver for the incidents CSV.
    Precedence:
      1) Environment variable INCIDENTS_CSV (must point to a file)
      2) The provided 'preferred' path (file or directory)
      3) Common filenames in common local folders (cwd, script dir, ./data, ../data)

    Returns: absolute path if found, else None.
    """
    # Determine the script directory up-front; all relative paths are resolved against this
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) Environment variable
    env_path = os.environ.get('INCIDENTS_CSV')
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    # Helper to try joining a directory with candidate names
    def try_candidates_in_dir(directory: str, names: list[str]):
        for name in names:
            candidate = os.path.abspath(os.path.join(directory, name))
            if os.path.isfile(candidate):
                return candidate
        return None

    # If preferred is provided
    if preferred:
        # If it's a relative path, treat it as relative to the script directory
        if not os.path.isabs(preferred):
            preferred_path = os.path.join(script_dir, preferred)
        else:
            preferred_path = preferred

        # Exact file path
        if os.path.isfile(preferred_path):
            return os.path.abspath(preferred_path)
        # If it's a directory, try candidates inside
        if os.path.isdir(preferred_path):
            candidates = [
                'incidents.csv',
            ]
            found = try_candidates_in_dir(preferred_path, candidates)
            if found:
                return found

    # 3) Common locations and filenames
    common_dirs = [
        os.path.join(script_dir, 'data'),  # Prioritize data directory
        os.getcwd(),
        script_dir,
        os.path.abspath(os.path.join(script_dir, '..', 'data')),
    ]
    common_names = [
        'incidents.csv',
    ]
    for d in common_dirs:
        found = try_candidates_in_dir(d, common_names)
        if found:
            return found

    # As a last lightweight attempt, look for any file that matches a loose pattern
    loose_patterns = [
        os.path.join(script_dir, 'data', '*.csv'),
        os.path.join(script_dir, '*.csv'),
        os.path.join(script_dir, 'data', '*.csv'),
    ]
    for pattern in loose_patterns:
        for path in glob.glob(pattern):
            base = os.path.basename(path).lower()
            if 'incident' in base or 'ransom' in base:
                return os.path.abspath(path)

    return None


def _calibration_assets_dir(script_dir: str) -> str:
    """Return the canonical calibration assets directory under output/.

    New location is output/calibration_assets. We'll maintain a read fallback to
    the legacy output/calibration directory for one release.
    """
    return os.path.join(script_dir, 'output', 'calibration_assets')


def _legacy_calibration_dir(script_dir: str) -> str:
    return os.path.join(script_dir, 'output', 'calibration')


def _load_calibration_method_recommendations(script_dir: str, comparison_file: str | None = None):
    """Load per-model calibration method recommendations from calibration_method_comparison.json.

    Returns a dict mapping model_config -> method ('histogram' or 'isotonic'). If the
    file is missing or invalid, returns an empty dict.
    """
    try:
        if comparison_file is None:
            comparison_file = os.path.join(_calibration_assets_dir(script_dir), 'calibration_method_comparison.json')
        if not os.path.isfile(comparison_file):
            # Fallback to legacy path once
            legacy = os.path.join(_legacy_calibration_dir(script_dir), 'calibration_method_comparison.json')
            comparison_file = legacy if os.path.isfile(legacy) else comparison_file
            if not os.path.isfile(comparison_file):
                return {}
        with open(comparison_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        per_model = data.get('per_model', []) or []
        mapping = {}
        for row in per_model:
            name = row.get('model_config')
            method = (row.get('recommended_method') or '').strip().lower()
            if name and method in ('histogram', 'isotonic'):
                mapping[name] = method
        return mapping
    except Exception:
        return {}

def _load_calibration_method_selection(script_dir: str, comparison_file: str | None = None):
    """Load per-model calibration method selection including method-specific params.

    Returns a dict: model_config -> { 'method': str, 'T': float|None, 's': float|None, 'source': path }
    """
    sel = {}
    try:
        if comparison_file is None:
            comparison_file = os.path.join(_calibration_assets_dir(script_dir), 'calibration_method_comparison.json')
        if not os.path.isfile(comparison_file):
            # Fallback to legacy path
            legacy = os.path.join(_legacy_calibration_dir(script_dir), 'calibration_method_comparison.json')
            comparison_file = legacy if os.path.isfile(legacy) else comparison_file
            if not os.path.isfile(comparison_file):
                return sel
        with open(comparison_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in (data.get('per_model') or []):
            name = row.get('model_config')
            if not name:
                continue
            method = (row.get('recommended_method') or '').strip().lower()
            T = None
            s = None
            if isinstance(row.get('temperature'), dict):
                T = row['temperature'].get('T')
            if isinstance(row.get('intensity'), dict):
                s = row['intensity'].get('s')
            sel[name] = {'method': method, 'T': T, 's': s, 'source': comparison_file}
    except Exception:
        pass
    return sel


def load_latest_data(data_file: str | None = None):
    """
    Loads the latest data from the specified CSV file (or resolves it automatically).
    In a real-world scenario, this would connect to a live data source.
    """
    resolved_path = find_data_file(preferred=data_file)
    if not resolved_path:
        print("[ERROR] Could not locate the incidents CSV.")
        print("        Set the environment variable INCIDENTS_CSV to the full path of your CSV,")
        print("        or place a file named one of the following in the script directory or ./data:")
        print("          - incidents.csv")
        return None
    print(f"Loading data from: {resolved_path}")
    try:
        # Try reading with 'utf-8', if that fails, try 'latin-1'
        try:
            df = pd.read_csv(resolved_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin-1...")
            df = pd.read_csv(resolved_path, encoding='latin-1')


        # Ensure 'Discovered Date' is datetime and set as index if needed later
        df['Discovered Date'] = pd.to_datetime(df['Discovered Date'])

        # Check for required columns before proceeding
        required_cols = ['Bundesland', 'Sector', 'Discovered Date']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"[ERROR] Data file is missing required columns: {missing_cols}")
            return None


        # Assuming 'week' is calculated and needed for time series analysis
        df['week'] = df['Discovered Date'].dt.to_period('W').apply(lambda r: r.start_time)
        return df
    except Exception as e:
        print(f"[ERROR] Error loading or processing data file: {e}")
        return None


def prepare_data(data_file, required_history_weeks, forecast_weeks):
    """
    Loads, cleans, and aggregates data to weekly counts per Target_V.
    Ensures enough data exists for the required history and forecast periods.
    """
    print(f"Preparing data from: {data_file}")
    df = load_latest_data(data_file)
    if df is None:
        return None, None, None, None

    # Define Target Node V: Bundesland x Sector (requires Bundesland column)
    # This check is now also in load_latest_data, but keeping here for clarity in prepare_data's logic
    if 'Bundesland' not in df.columns or 'Sector' not in df.columns or 'Discovered Date' not in df.columns:
         # This case should ideally be caught by load_latest_data now
        raise ValueError("Data file must contain 'Bundesland', 'Sector', and 'Discovered Date' columns.")

    # Normalize dash-like characters in Sector names to a standard hyphen and ensure lower-case
    def _normalize_dashes(text: str) -> str:
        if pd.isna(text):
            return text
        # Replace various dash-like unicode/control characters with ascii hyphen
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\u0096\u00ad]", "-", str(text))
        # Collapse repeated whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df['Sector'] = df['Sector'].fillna('other').apply(_normalize_dashes).str.lower()
    # Also normalize Bundesland dashes to be safe (preserve original case otherwise)
    df['Bundesland'] = df['Bundesland'].apply(lambda x: re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\u0096\u00ad]", "-", str(x)).strip())

    df['Target_V'] = df['Bundesland'] + ' | ' + df['Sector']


    # Weekly counts per Target Node V
    weekly_counts_v = df.groupby(['week', 'Target_V']).size().reset_index(name='count')
    Y_t = weekly_counts_v.pivot_table(index='week', columns='Target_V', values='count').fillna(0)

    # Ensure the index is a DatetimeIndex for time-based slicing
    Y_t.index = pd.to_datetime(Y_t.index)


    num_targets = len(Y_t.columns)
    total_weeks = len(Y_t)

    # Check if enough data exists
    required_total_weeks = required_history_weeks + forecast_weeks
    if total_weeks < required_total_weeks:
        print(f"[WARNING] Not enough data for required history ({required_history_weeks}) + forecast ({forecast_weeks}) weeks. Total weeks available: {total_weeks}. Required: {required_total_weeks}.")
        # Decide how to handle: raise error, use less data, etc.
        # For now, let's proceed with available data but print warning.
        # In a real system, you might want to raise an error or log this condition.


    print(f"Data Prepared. Targets (V): {num_targets}. Total weeks in data: {total_weeks}.")
    return Y_t, num_targets, total_weeks, Y_t.index # Return Y_t index for later use in date calculations


# Define calculate_log_loss as it's a utility function for backtesting/optimization
def calculate_log_loss(y_true, p_pred):
    """
    Log Loss / Negative Log Likelihood (NLL) for a single forecast step.
    y_true: Binary indicator of event occurrence (0 or 1).
    p_pred: Predicted probability of event (P(count >= 1)).
    """
    # Clip predictions to prevent log(0)
    p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)

    # Log loss for a Bernoulli outcome
    # Ensure y_true and p_pred are numpy arrays for element-wise operations
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    nll = -np.sum(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
    return nll

"""### 3. Model Forecasting Functions

This section contains the core forecasting logic for the Baseline, Contagion-Only, and Hybrid models.
"""

# Assume DECAY_FIXED and JUMP_FIXED are defined globally or passed as parameters
# Consistent Parameters (can be overridden by optimization)
DECAY_FIXED = 0.9
JUMP_FIXED = 0.05

def baseline_forecast(Y_train):
    """Model 1: Historical Frequency Baseline (Static Strategic)."""
    # Strategic Baseline Intensity (mu_v)
    # Calculate mean over the training window
    lambda_v = Y_train.mean(axis=0)
    # Convert intensity to probability of at least one event (P(count >= 1))
    P_base = 1 - np.exp(-lambda_v)
    return P_base

def contagion_only_forecast(Y_train, Y_current, H_t_prev, decay, jump):
    """Model 2: Contagion-Only (Global Mean + Hawkes H_v(t))."""
    # Global Baseline Intensity (mu_global) - calculated from Y_train
    # In a real-time forecast, Y_train would be the latest training window
    mu_global = Y_train.values.mean()

    # Contagion/Hawkes Component Update
    # H_t_prev is the Hawkes memory *before* the current observation Y_current
    # The new Hawkes memory H_t is updated using the observed events Y_current
    H_t = decay * H_t_prev + jump * Y_current

    # Total Intensity is Global Baseline + Hawkes Component
    lambda_v = mu_global + H_t
    # Convert intensity to probability of at least one event
    P_contagion = 1 - np.exp(-lambda_v)

    # Return forecast probability for the *next* step and the updated Hawkes memory
    return P_contagion, H_t

def hybrid_forecast(Y_train, Y_current, H_t_prev, decay, jump):
    """Model 3: Hybrid (Target-Specific Strategic mu_v + Hawkes H_v(t))."""
    # Strategic Baseline Intensity (mu_v) - target-specific average rate
    # Calculated from the training window Y_train
    mu_v = Y_train.mean(axis=0).values

    # Contagion/Hawkes Component Update
    # H_t_prev is the Hawkes memory *before* the current observation Y_current
    # The new Hawkes memory H_t is updated using the observed events Y_current
    H_t = decay * H_t_prev + jump * Y_current

    # Total Intensity is Target-Specific Baseline + Hawkes Component
    lambda_v = mu_v + H_t
    # Convert intensity to probability of at least one event
    P_hybrid = 1 - np.exp(-lambda_v)

    # Return forecast probability for the *next* step and the updated Hawkes memory
    return P_hybrid, H_t

"""### 4. Backtesting Framework

This section contains the function used for rolling backtesting of the models and parameter optimization.
"""

# Assume prepare_data, calculate_log_loss, baseline_forecast,
# contagion_only_forecast, hybrid_forecast, DECAY_FIXED, and JUMP_FIXED
# are already defined in previous cells.

def do_backtest(model_name: str,
                data_file: str | None = None,
                decay_values: np.ndarray = np.arange(0.1, 1.0, 0.05), # Expanded decay range for optimization sweep
                jump_values: np.ndarray = np.arange(0.001, 0.201, 0.01), # Expanded jump range for optimization sweep
                T_TRAIN_OPT: int = 8, # Default optimization training window (overridden by config)
                T_TEST_OPT: int = 1, # Default optimization test window (overridden by config)
               ):
    """
    Runs a rolling backtest for specified models against the historical data.
    Includes an inner optimization loop for DECAY_FIXED and JUMP_FIXED if
    performing parameter sweep, using configuration-specific optimization windows.

    :param model_name: 'baseline', 'contagion', 'hybrid', or 'all'.
    :param data_file: Optional path to the incidents CSV. If None, a resolver will search common locations.
    :param decay_values: Array of potential values for DECAY_FIXED in the optimization sweep. Defaults to np.arange(0.1, 1.0, 0.05).
    :param jump_values: Array of potential values for JUMP_FIXED in the optimization sweep. Defaults to np.arange(0.001, 0.201, 0.01).
    :param T_TRAIN_OPT: The number of weeks for the optimization training window. Defaults to 8. (These defaults are overridden by config)
    :param T_TEST_OPT: The number of weeks for the optimization test period. Defaults to 1. (These defaults are overridden by config)
    :return: A DataFrame with Log Loss results.
    """

    # Define model configurations with their specific backtest and optimal optimization windows
    model_configs = {}

    # Define Baseline configurations for each relevant backtest window
    baseline_configs = {
        'baseline (8/1)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 8, 'T_TEST_BACKTEST': 1, 'optimize_params': False},
        'baseline (26/13)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 26, 'T_TEST_BACKTEST': 13, 'optimize_params': False},
        'baseline (52/26)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 52, 'T_TEST_BACKTEST': 26, 'optimize_params': False},
    }


    if model_name.lower() in ['all', 'baseline']:
        model_configs.update(baseline_configs) # Add all baselines if 'baseline' or 'all' is requested

    if model_name.lower() in ['all', 'contagion']:
        # Contagion specific configuration with optimal optimization window
        model_configs['contagion (8/1 Opt)'] = {
            'func': contagion_only_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 8,
            'T_TEST_BACKTEST': 1,
            'optimize_params': True,
            'T_TRAIN_OPT': 4, # Optimal T_TRAIN_OPT found
            'T_TEST_OPT': 1   # Optimal T_TEST_OPT for Contagion (8/1)
        }
    if model_name.lower() in ['all', 'hybrid']:
        # Hybrid specific configurations with their optimal optimization windows
        model_configs['hybrid (26/13 Opt)'] = {
            'func': hybrid_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 26,
            'T_TEST_BACKTEST': 13,
            'optimize_params': True,
            'T_TRAIN_OPT': 16, # Optimal T_TRAIN_OPT found
            'T_TEST_OPT': 4    # Optimal T_TEST_OPT for Hybrid (26/13)
        }
        model_configs['hybrid (52/26 Opt)'] = {
            'func': hybrid_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 52,
            'T_TEST_BACKTEST': 26,
            'optimize_params': True,
            'T_TRAIN_OPT': 4, # Optimal T_TRAIN_OPT found
            'T_TEST_OPT': 1    # Optimal T_TEST_OPT for Hybrid (52/26)
        }


    if not model_configs:
        print(f"[ERROR] Invalid model_name '{model_name}'. Choose 'baseline', 'contagion', 'hybrid', or 'all'.")
        return None

    # Load data once based on the maximum required data length across all configurations
    max_train_backtest = max([config['T_TRAIN_BACKTEST'] for config in model_configs.values()])
    max_test_backtest = max([config['T_TEST_BACKTEST'] for config in model_configs.values()])
    # Determine max optimization window needed across all configs that optimize
    max_train_opt_needed = max([config.get('T_TRAIN_OPT', 0) for config in model_configs.values() if config.get('optimize_params', False)])
    max_test_opt_needed = max([config.get('T_TEST_OPT', 0) for config in model_configs.values() if config.get('optimize_params', False)])


    # Ensure enough data for the longest backtest window including the optimization period preceding the very first forecast step
    # The data needs to cover from the earliest point required by any config's first forecast step
    # up to the latest point required by any config's last forecast step.
    # The earliest point is min(actual_start_index for all configs) - max_train_backtest
    # The latest point is max(end_index_cfg for all configs)
    earliest_start_offset = max(max_train_backtest, max_train_opt_needed + max_test_opt_needed) # The minimum number of weeks required before the first forecast step
    latest_end_index = max([config['T_TRAIN_BACKTEST'] + config['T_TEST_BACKTEST'] for config in model_configs.values()])


    try:
        # Prepare data with enough history for all configurations
        Y_t_full, _, _, Y_t_index = prepare_data(data_file, earliest_start_offset, latest_end_index)
        num_targets = len(Y_t_full.columns)

    except Exception as e:
        print(f"[ERROR] Data Preparation Failed: {e}")
        return None

    all_results = []

    for config_name, config in model_configs.items():
        func = config['func']
        T_TRAIN_BACKTEST_CFG = config['T_TRAIN_BACKTEST']
        T_TEST_BACKTEST_CFG = config['T_TEST_BACKTEST']
        optimize_params_cfg = config['optimize_params']
        H_t_prev = np.zeros(num_targets) if config_name.startswith(('contagion', 'hybrid')) else None # Initialize Hawkes memory for non-baseline models


        # Get config-specific optimization windows, use defaults if not specified
        # These are only used if optimize_params_cfg is True
        current_T_TRAIN_OPT = config.get('T_TRAIN_OPT', T_TRAIN_OPT)
        current_T_TEST_OPT = config.get('T_TEST_OPT', T_TEST_OPT)


        # Adjust start and end index based on the current config's backtest window
        # Ensure enough data for both the main training window and the optimization window
        # The backtest for this config starts after its training window
        start_index_cfg = T_TRAIN_BACKTEST_CFG
        end_index_cfg = T_TRAIN_BACKTEST_CFG + T_TEST_BACKTEST_CFG

        # If optimization is enabled, the actual start of the loop must also account for the optimization window
        if optimize_params_cfg:
             actual_start_index = max(start_index_cfg, current_T_TRAIN_OPT + current_T_TEST_OPT)
        else:
             actual_start_index = start_index_cfg


        # Check if enough data exists for this specific configuration including the optimization window
        if len(Y_t_full) < end_index_cfg:
             print(f"[WARNING] Not enough data for configuration '{config_name}'. Required backtest length: {end_index_cfg}. Available: {len(Y_t_full)}. Skipping.")
             continue

        if len(Y_t_full) < actual_start_index + 1: # Check if the actual start index is within the data bounds
             print(f"[WARNING] Not enough data to start backtest for configuration '{config_name}' with optimization. Required start index: {actual_start_index}. Available data points: {len(Y_t_full)}. Skipping.")
             continue


        # Slice the data for the current configuration's relevant period (up to the end of its backtest)
        # Y_t = Y_t_full.iloc[:end_index_cfg] # No longer needed, use Y_t_full directly with adjusted indices

        log_losses = []

        print(f"\nStarting Rolling Backtest for {config_name} (Total {T_TEST_BACKTEST_CFG} weeks, {num_targets} targets)...")
        print(f"  Main Training Window: {T_TRAIN_BACKTEST_CFG}, Main Test Window: {T_TEST_BACKTEST_CFG}")
        if optimize_params_cfg:
             print(f"  Optimization Training Window: {current_T_TRAIN_OPT}, Optimization Test Window: {current_T_TEST_OPT}")
             print(f"  Actual Backtest Start Index (including opt window): {actual_start_index}")


        # The backtest loop range should be inclusive of the last index for which a forecast is needed
        # The loop now iterates over the indices of the full dataset Y_t_full
        for t in range(actual_start_index, end_index_cfg):
            # The training data for the main backtest step 't' is the T_TRAIN_BACKTEST_CFG weeks *before* t
            Y_train = Y_t_full.iloc[t - T_TRAIN_BACKTEST_CFG : t]
            Y_obs_counts = Y_t_full.iloc[t] # Observation for the current step
            y_true_binary = (Y_obs_counts > 0).astype(int)


            if config_name.startswith('baseline'):
                P_forecast = func(Y_train)
                log_losses.append(calculate_log_loss(y_true_binary.values, P_forecast.values))
            else:
                optimal_decay = DECAY_FIXED # Default to global fixed if no optimization
                optimal_jump = JUMP_FIXED   # Default to global fixed if no optimization

                if optimize_params_cfg:
                    best_nll_for_step = float('inf')

                    # Define optimization period data - relative to the full dataset Y_t_full
                    # The optimization window immediately precedes the current step 't'
                    opt_start_index_full = t - current_T_TEST_OPT - current_T_TRAIN_OPT
                    opt_end_index_full = t - current_T_TEST_OPT # The optimization is performed on data up to t - T_TEST_OPT - 1

                    if opt_start_index_full < 0:
                         # This case should be handled by adjusting the actual_start_index,
                         # but this check is a safeguard/debug print.
                         # With the correct actual_start_index, this block should not be reached during the loop.
                        print(f"[ERROR] Insufficient data for optimization at step {t} for {config_name}. This should not happen with corrected actual_start_index.")
                        # Should not continue if this error is hit, but keeping the 'pass' for now
                        pass
                    else:
                        # The mini-backtest for optimization runs over T_TEST_OPT weeks
                        # The training data for the mini-backtest steps comes from Y_t_full using offsets from opt_start_index_full

                        for decay in decay_values:
                            for jump in jump_values:
                                # Perform mini-backtest over the optimization period
                                current_H_prev_mini = np.zeros(num_targets)
                                mini_log_losses = []

                                # Iterate over the optimization test period
                                for opt_t_relative in range(current_T_TRAIN_OPT, current_T_TRAIN_OPT + current_T_TEST_OPT):
                                    opt_t_full = opt_start_index_full + opt_t_relative # Index in Y_t_full

                                    Y_train_mini = Y_t_full.iloc[opt_t_full - current_T_TRAIN_OPT : opt_t_full]
                                    Y_obs_counts_mini = Y_t_full.iloc[opt_t_full].values # Get numpy array
                                    y_true_binary_mini = (Y_obs_counts_mini > 0).astype(int)

                                    P_forecast_mini, H_t_new_mini = func(
                                        Y_train=Y_train_mini,
                                        Y_current=Y_obs_counts_mini,
                                        H_t_prev=current_H_prev_mini,
                                        decay=decay,
                                        jump=jump
                                    )
                                    mini_log_losses.append(calculate_log_loss(y_true_binary_mini, P_forecast_mini))
                                    current_H_prev_mini = H_t_new_mini

                                # Ensure mini_log_losses is not empty before calculating mean
                                if mini_log_losses:
                                     avg_nll_mini = np.mean(mini_log_losses)
                                else:
                                     # If mini_backtest had no steps, this parameter combo is invalid for this window
                                     avg_nll_mini = float('inf') # Assign a very high NLL

                                if avg_nll_mini < best_nll_for_step:
                                    best_nll_for_step = avg_nll_mini
                                    optimal_decay = decay
                                    optimal_jump = jump

                # Use optimal parameters for the current week's forecast (t) in the main backtest
                P_forecast, H_t_new = func(
                    Y_train=Y_train, # Use the current main training window slice
                    Y_current=Y_obs_counts,
                    H_t_prev=H_t_prev,
                    decay=optimal_decay,
                    jump=optimal_jump
                )
                log_losses.append(calculate_log_loss(y_true_binary.values, P_forecast))
                H_t_prev = H_t_new # Update Hawkes memory for next main step

        # --- Final Comparison and Output for this config ---
        if log_losses: # Only calculate if there were successful forecast steps
            avg_log_loss = np.mean(log_losses)

            all_results.append({
                'Model': config_name, # Store raw config name for easier processing later
                'Average Log Loss (NLL)': avg_log_loss,
                # Store backtest window for easier skill score calculation
                'T_TRAIN_BACKTEST': T_TRAIN_BACKTEST_CFG,
                'T_TEST_BACKTEST': T_TEST_BACKTEST_CFG,
                'Skill Score vs. Baseline (%)': 0.0 # Placeholder, will be calculated later
            })
        else:
             print(f"[WARNING] No successful forecast steps for configuration '{config_name}'. No results recorded.")


    # --- Final Comparison and Output for all configs ---
    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        # Calculate Skill Score after all models have run and baseline NLL is available
        # Store baseline NLLs in a dictionary keyed by their backtest window tuple
        baseline_nlls = {}
        # Ensure baseline configs are processed first or handle potential missing keys
        for bl_config_name in baseline_configs.keys(): # Use the defined baseline_configs
             bl_row = results_df[results_df['Model'] == bl_config_name]
             if not bl_row.empty:
                  baseline_nlls[(bl_row['T_TRAIN_BACKTEST'].iloc[0], bl_row['T_TEST_BACKTEST'].iloc[0])] = bl_row['Average Log Loss (NLL)'].iloc[0]
             else:
                  print(f"[WARNING] Baseline config '{bl_config_name}' results not found. Cannot calculate Skill Scores relative to this baseline.")
                  # baseline_nlls[corresponding window tuple] = np.nan # Or handle missing baseline appropriately


        results_df['Skill Score vs. Baseline (%)'] = results_df.apply(
            lambda row: (baseline_nlls.get((row['T_TRAIN_BACKTEST'], row['T_TEST_BACKTEST']), np.nan) - row['Average Log Loss (NLL)']) / baseline_nlls.get((row['T_TRAIN_BACKTEST'], row['T_TEST_BACKTEST']), np.nan) * 100 if not row['Model'].startswith('baseline') and (row['T_TRAIN_BACKTEST'], row['T_TEST_BACKTEST']) in baseline_nlls else 0.0 if row['Model'].startswith('baseline') else np.nan,
            axis=1
        )

        # Drop the temporary backtest window columns
        # results_df = results_df.drop(columns=['T_TRAIN_BACKTEST', 'T_TEST_BACKTEST']) # Keep these columns for now if needed for later processing


        results_df['Model'] = results_df['Model'].replace({
            'baseline (8/1)': '1a. Historical Frequency Baseline (Static) (8/1)',
            'baseline (26/13)': '1b. Historical Frequency Baseline (Static) (26/13)',
            'baseline (52/26)': '1c. Historical Frequency Baseline (Static) (52/26)',
            'contagion (8/1 Opt)': '2. Contagion-Only Model (8/1 Opt)',
            'hybrid (26/13 Opt)': '3a. Hybrid Model (26/13 Opt)',
            'hybrid (52/26 Opt)': '3b. Hybrid Model (52/26 Opt)',
        })
        results_df = results_df.sort_values('Average Log Loss (NLL)')

    print("\n--- Final Backtest Results ---")
    # print(results_df.to_string(index=False)) # Use display for better formatting
    display(results_df)

    return results_df

"""### 5. Daily Forecasting Implementation

This section contains the function to run daily forecasts using the optimized models and configurations, and defines the specific configurations to be used for forecasting.
"""

# Assume load_latest_data, prepare_data, baseline_forecast,
# contagion_only_forecast, hybrid_forecast, calculate_log_loss,
# DECAY_FIXED, and JUMP_FIXED are already defined in previous cells.

# Define model configurations to be used for daily forecasting
# These include the determined optimal optimization windows
model_forecast_configs = {
    'contagion (8/1 Opt)': {
        'T_TRAIN_BACKTEST': 8,
        'T_TEST_BACKTEST': 1,
        'T_TRAIN_OPT': 4, # Optimal T_TRAIN_OPT found
        'T_TEST_OPT': 1,  # Optimal T_TEST_OPT found
        'decay_values': np.arange(0.1, 1.0, 0.05), # Expanded parameter ranges used in the sweep
        'jump_values': np.arange(0.001, 0.201, 0.01), # Expanded parameter ranges used in the sweep
        'func': contagion_only_forecast,
        'H_prev': None # Placeholder, will be initialized later
    },
    'hybrid (26/13 Opt)': {
        'T_TRAIN_BACKTEST': 26,
        'T_TEST_BACKTEST': 13,
        'T_TRAIN_OPT': 16, # Optimal T_TRAIN_OPT found
        'T_TEST_OPT': 4,  # Optimal T_TEST_OPT found
        'decay_values': np.arange(0.1, 1.0, 0.05), # Expanded parameter ranges used in the sweep
        'jump_values': np.arange(0.001, 0.201, 0.01), # Expanded parameter ranges used in the sweep
        'func': hybrid_forecast,
        'H_prev': None # Placeholder, will be initialized later
    },
    'hybrid (52/26 Opt)': {
        'T_TRAIN_BACKTEST': 52,
        'T_TEST_BACKTEST': 26,
        'T_TRAIN_OPT': 4, # Optimal T_TRAIN_OPT found
        'T_TEST_OPT': 1,  # Optimal T_TEST_OPT found
        'decay_values': np.arange(0.1, 1.0, 0.05), # Expanded parameter ranges used in the sweep
        'jump_values': np.arange(0.001, 0.201, 0.01), # Expanded parameter ranges used in the sweep
        'func': hybrid_forecast,
        'H_prev': None # Placeholder, will be initialized later
    }
}


# === Calibration helpers (histogram binning using saved bins files) ===
def _model_config_to_slug(model_config_name: str) -> str:
    """Map a model_config to the slug used by calibration artifacts.

    Expected mappings:
      contagion (8/1 Opt) -> contagion_8-1_opt
      hybrid (26/13 Opt)  -> hybrid_26-13_opt
      hybrid (52/26 Opt)  -> hybrid_52-26_opt
    """
    name = model_config_name.strip().lower()
    mapping = {
        'contagion (8/1 opt)': 'contagion_8-1_opt',
        'hybrid (26/13 opt)': 'hybrid_26-13_opt',
        'hybrid (52/26 opt)': 'hybrid_52-26_opt',
    }
    return mapping.get(name, ''.join(ch for ch in name if ch.isalnum() or ch in ['_', '-', '/']).replace(' ', '_'))


def _load_calibration_bins(model_config_name: str, script_dir: str, calibration_dir: str = None):
    """Load calibration bins (centers, event rates, and counts) from output/calibration for the given model.

    Returns (centers: np.ndarray, event_rates: np.ndarray, bin_counts: np.ndarray, path: str)
    or (None, None, None, None) if unavailable.
    """
    if calibration_dir is None:
        calibration_dir = _calibration_assets_dir(script_dir)
    slug = _model_config_to_slug(model_config_name)
    bins_file = os.path.join(calibration_dir, f"bins_{slug}.csv")
    if not os.path.isfile(bins_file):
        # Fallback to legacy location once
        legacy_bins = os.path.join(_legacy_calibration_dir(script_dir), f"bins_{slug}.csv")
        if os.path.isfile(legacy_bins):
            bins_file = legacy_bins
        else:
            return None, None, None, None
    try:
        bins_df = pd.read_csv(bins_file)
        # Expect columns: bin_center, bin_count, mean_pred, event_rate
        centers = bins_df['bin_center'].to_numpy(dtype=float)
        event_rates = bins_df['event_rate'].to_numpy(dtype=float)
        bin_counts = bins_df['bin_count'].to_numpy(dtype=float) if 'bin_count' in bins_df.columns else np.ones_like(event_rates)
        # Filter out bins with zero count to avoid degenerate mapping if present
        if 'bin_count' in bins_df.columns:
            mask = bins_df['bin_count'].to_numpy(dtype=int) > 0
            if mask.any():
                centers = centers[mask]
                event_rates = event_rates[mask]
                bin_counts = bin_counts[mask]
        return centers, event_rates, bin_counts, bins_file
    except Exception:
        return None, None, None, None


def _apply_histogram_calibration(probs: np.ndarray, centers: np.ndarray, event_rates: np.ndarray) -> np.ndarray:
    """Calibrate probabilities by mapping to nearest bin center's observed event rate.

    Vectorized nearest-neighbor on bin centers. If inputs are 2D, treat last axis as targets.
    """
    p = np.asarray(probs)
    if centers is None or event_rates is None or centers.size == 0:
        return p  # identity
    # Flatten last axis for vectorized operation, then reshape back
    orig_shape = p.shape
    flat = p.reshape(-1)
    # Compute nearest center index for each prob
    # Shape (N, B)
    diff = np.abs(flat[:, None] - centers[None, :])
    idx = np.argmin(diff, axis=1)
    calibrated = event_rates[idx]
    return calibrated.reshape(orig_shape)


def _apply_laplace_smoothing(event_rates: np.ndarray, bin_counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Apply Laplace smoothing to event rates estimated from bins.

    event_rates: observed event frequency in bin (events/bin_counts)
    bin_counts: number of samples in each bin
    alpha: pseudo count for both positive and negative classes (beta prior with a=b=alpha)

    Returns smoothed event rate per bin.
    """
    try:
        events = np.clip(event_rates, 0.0, 1.0) * np.clip(bin_counts, 0.0, None)
        smoothed = (events + alpha) / (np.clip(bin_counts, 0.0, None) + 2.0 * alpha)
        return np.clip(smoothed, 0.0, 1.0)
    except Exception:
        return event_rates


def _fit_isotonic_mapping(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Fit an isotonic (non-decreasing) mapping y_hat(x) using the PAV algorithm on binned points.

    x: bin centers (1D), y: target event rates (1D), w: weights (bin counts).
    Returns x_sorted, y_iso (both 1D arrays with the same length), representing a stepwise non-decreasing fit.
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if w is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(w).astype(float)

    # Sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]

    # Pool Adjacent Violators
    y_hat = y.copy()
    weights = w.copy()
    n = len(y_hat)
    i = 0
    while i < n - 1:
        if y_hat[i] <= y_hat[i + 1] + 1e-12:
            i += 1
            continue
        # merge blocks until isotonicity holds
        j = i
        while j >= 0 and y_hat[j] > y_hat[j + 1] + 1e-12:
            total_w = weights[j] + weights[j + 1]
            if total_w > 0:
                avg = (weights[j] * y_hat[j] + weights[j + 1] * y_hat[j + 1]) / total_w
            else:
                avg = (y_hat[j] + y_hat[j + 1]) / 2.0
            y_hat[j] = avg
            y_hat[j + 1] = avg
            weights[j] = total_w
            # collapse by shifting left
            if j + 1 < n - 1:
                y_hat = np.delete(y_hat, j + 1)
                x = np.delete(x, j + 1)
                weights = np.delete(weights, j + 1)
                n -= 1
            j -= 1
        i = max(j + 1, 0)

    return x, np.clip(y_hat, 0.0, 1.0)


def _apply_isotonic(probs: np.ndarray, x_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """Apply isotonic mapping using piecewise linear interpolation over the fitted points."""
    p = np.asarray(probs)
    orig_shape = p.shape
    flat = p.reshape(-1)
    # Ensure boundaries are covered
    x_fit = np.asarray(x_fit).astype(float)
    y_fit = np.asarray(y_fit).astype(float)
    if x_fit.size < 2:
        return p  # not enough points
    # clip flat into [min(x), max(x)] and interpolate
    flat_clipped = np.clip(flat, x_fit[0], x_fit[-1])
    calibrated = np.interp(flat_clipped, x_fit, y_fit)
    return calibrated.reshape(orig_shape)


def run_daily_forecast(data_file, model_configs):
    """
    Runs the daily forecasting process for configured models, including
    parameter optimization and forecast generation. Returns raw forecast
    probabilities and average weekly probabilities.

    Args:
        data_file (str): Path to the latest data CSV file.
        model_configs (dict): Dictionary of model configurations.

    Returns:
        list: A list of dictionaries, each containing forecast results for a model,
              including 'forecast_probabilities' and 'average_weekly_probabilities'.
    """
    print(f"--- Starting Daily Forecast Process ({pd.Timestamp.now()}) ---")

    # Load and prepare the data using the prepare_data function
    # Determine required history for prepare_data based on max config window
    max_train_backtest = max([config['T_TRAIN_BACKTEST'] for config in model_configs.values()])
    max_test_backtest = max([config['T_TEST_BACKTEST'] for config in model_configs.values()])
    max_train_opt_needed = max([config.get('T_TRAIN_OPT', 0) for config in model_configs.values()])
    max_test_opt_needed = max([config.get('T_TEST_OPT', 0) for config in model_configs.values()])
    earliest_start_offset = max(max_train_backtest, max_train_opt_needed + max_test_opt_needed)
    latest_end_index = max([config['T_TRAIN_BACKTEST'] + config['T_TEST_BACKTEST'] for config in model_configs.values()])


    Y_t, num_targets, total_weeks, Y_t_index = prepare_data(data_file, earliest_start_offset, latest_end_index)

    if Y_t is None:
        print("[ERROR] Failed to prepare data. Aborting daily forecast.")
        return []


    all_forecast_results = []

    # Determine the end index of the available data (latest week)
    latest_week_index = total_weeks - 1

    # Determine whether to apply post-calibration using saved calibration bins
    apply_calibration = os.environ.get('APPLY_CALIBRATION', 'true').lower() != 'false'
    cal_method_env = os.environ.get('CALIBRATION_METHOD', '').strip().lower()
    cal_apply_smoothing = os.environ.get('CAL_APPLY_SMOOTHING', 'true').lower() != 'false'
    try:
        cal_laplace_alpha = float(os.environ.get('CAL_LAPLACE_ALPHA', '0.5'))
    except Exception:
        cal_laplace_alpha = 0.5
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Load per-model recommended calibration methods and parameters if no global override
    selection_map = {}
    if not cal_method_env:
        selection_map = _load_calibration_method_selection(script_dir)

    for config_name, config in model_configs.items():
        print(f"\nProcessing model configuration: {config_name}")

        func = config['func']
        T_TRAIN_BACKTEST_CFG = config['T_TRAIN_BACKTEST']
        T_TEST_BACKTEST_CFG = config['T_TEST_BACKTEST']
        # optimize_params_cfg is always True for these configs
        optimize_params_cfg = True # Force optimization for these forecast configs
        current_T_TRAIN_OPT = config.get('T_TRAIN_OPT', 8) # Use default if not in config
        current_T_TEST_OPT = config.get('T_TEST_OPT', 1)  # Use default if not in config
        decay_values = config.get('decay_values', np.arange(0.5, 1.0, 0.1))
        jump_values = config.get('jump_values', np.arange(0.01, 0.1, 0.02))

        # Determine the start index for the main training window based on the latest data
        main_train_start_index = latest_week_index - T_TRAIN_BACKTEST_CFG + 1
        main_train_end_index = latest_week_index + 1 # End index is exclusive

        # Check if enough data is available for the main training window
        if main_train_start_index < 0:
            print(f"[WARNING] Not enough data for main training window ({T_TRAIN_BACKTEST_CFG} weeks) for {config_name}. Required start index: {main_train_start_index}. Available data points: {total_weeks}. Skipping.")
            continue

        Y_train_main = Y_t.iloc[main_train_start_index : main_train_end_index]

        optimal_decay = DECAY_FIXED # Default to global fixed
        optimal_jump = JUMP_FIXED   # Default to global fixed

        # Optimization is always performed for these models
        best_nll_for_opt = float('inf')

        # Define optimization period data - immediately preceding the main training window
        opt_start_index_full = main_train_start_index - current_T_TEST_OPT - current_T_TRAIN_OPT
        opt_end_index_full = main_train_start_index - current_T_TEST_OPT # Optimization is performed up to opt_end_index_full - 1

        # Check if enough data is available for the optimization window
        if opt_start_index_full < 0:
            print(f"[WARNING] Not enough data for optimization window (Train: {current_T_TRAIN_OPT}, Test: {current_T_TEST_OPT}) for {config_name}. Required start index: {opt_start_index_full}. Available data points: {total_weeks}. Using default parameters.")
            # Continue with default parameters
            pass
        else:
            # Optimization data slice for the mini-backtest
            Y_opt_period_slice = Y_t.iloc[opt_start_index_full : main_train_start_index]

            if len(Y_opt_period_slice) < current_T_TRAIN_OPT + current_T_TEST_OPT:
                print(f"[WARNING] Optimization data slice too short ({len(Y_opt_period_slice)} weeks) for window (Train: {current_T_TRAIN_OPT}, Test: {current_T_TEST_OPT}) for {config_name}. Using default parameters.")
                # Continue with default parameters
                pass
            else:
                print(f"Performing optimization sweep for {config_name} over optimization window...")
                # Perform mini-backtest over the optimization period to find best parameters
                # The mini-backtest runs over the last T_TEST_OPT weeks of the optimization slice
                mini_backtest_start_relative = current_T_TRAIN_OPT
                mini_backtest_end_relative = current_T_TRAIN_OPT + current_T_TEST_OPT


                for decay in decay_values:
                    for jump in jump_values:
                        current_H_prev_mini = np.zeros(num_targets)
                        mini_log_losses = []

                        # Iterate over the optimization test period (relative to the start of Y_opt_period_slice)
                        for opt_t_relative in range(mini_backtest_start_relative, mini_backtest_end_relative):
                            opt_t_slice_index = opt_t_relative # Index within Y_opt_period_slice

                            Y_train_mini = Y_opt_period_slice.iloc[opt_t_slice_index - current_T_TRAIN_OPT : opt_t_slice_index]
                            Y_obs_counts_mini = Y_opt_period_slice.iloc[opt_t_slice_index].values # Get numpy array
                            y_true_binary_mini = (Y_obs_counts_mini > 0).astype(int)

                            P_forecast_mini, H_t_new_mini = func(
                                Y_train=Y_train_mini,
                                Y_current=Y_obs_counts_mini,
                                H_t_prev=current_H_prev_mini,
                                decay=decay,
                                jump=jump
                            )
                            mini_log_losses.append(calculate_log_loss(y_true_binary_mini, P_forecast_mini))
                            current_H_prev_mini = H_t_new_mini

                        if mini_log_losses:
                             avg_nll_mini = np.mean(mini_log_losses)
                        else:
                             avg_nll_mini = float('inf') # Assign a very high NLL if mini-backtest was empty

                        if avg_nll_mini < best_nll_for_opt:
                            best_nll_for_step = avg_nll_mini
                            optimal_decay = decay
                            optimal_jump = jump
                print(f"Optimization complete. Optimal parameters: DECAY={optimal_decay:.2f}, JUMP={optimal_jump:.2f}")


        # --- Train model on the full main training data with optimal/default parameters ---
        # Initialize Hawkes memory for the start of the main training window
        H_prev_main = np.zeros(num_targets) # These models use Hawkes

        # Iterate through the main training window to train the model and update H_prev
        for train_t_relative in range(T_TRAIN_BACKTEST_CFG):
             train_t_full = main_train_start_index + train_t_relative # Index in Y_t

             Y_train_slice = Y_t.iloc[train_t_full - T_TRAIN_BACKTEST_CFG : train_t_full] # Not strictly needed for func, but good practice
             Y_obs_counts_train = Y_t.iloc[train_t_full].values # Get numpy array

             # Train step: update H_prev using observed data and optimal parameters
             _, H_prev_main = func(
                 Y_train=Y_train_slice, # Pass a slice, though not used by func for training
                 Y_current=Y_obs_counts_train,
                 H_t_prev=H_prev_main,
                 decay=optimal_decay,
                 jump=optimal_jump
             )


        # --- Generate Forecast for the next T_TEST_BACKTEST_CFG weeks ---
        # The index of Y_t is now DatetimeIndex, so addition with Timedelta is valid
        forecast_start_date = Y_t.index[latest_week_index] + pd.Timedelta(weeks=1)
        forecast_horizon = T_TEST_BACKTEST_CFG

        # The forecast is generated for the period immediately following the latest data point
        # The forecast for week `latest_week_index + 1` uses the state (H_prev_main)
        # from the end of the training window (`latest_week_index`).

        forecast_probabilities = []
        # Prepare calibration artifacts for this model
        calib_centers, calib_rates, calib_counts, calib_path = (None, None, None, None)
        comparison_path = os.path.join(_calibration_assets_dir(script_dir), 'calibration_method_comparison.json')
        if not os.path.isfile(comparison_path):
            # fallback to legacy for one release
            legacy_cmp = os.path.join(_legacy_calibration_dir(script_dir), 'calibration_method_comparison.json')
            if os.path.isfile(legacy_cmp):
                comparison_path = legacy_cmp
        if apply_calibration:
            # For histogram/isotonic we need bins; for temperature/intensity we don't
            calib_centers, calib_rates, calib_counts, calib_path = _load_calibration_bins(config_name, script_dir)
        current_H_forecast = H_prev_main # Start forecasting with the H_prev from the end of training

        # Determine calibration method and params
        allowed_methods = ('histogram', 'isotonic', 'temperature', 'intensity')
        chosen_method = None
        chosen_T = None
        chosen_s = None
        if apply_calibration:
            if cal_method_env in allowed_methods:
                chosen_method = cal_method_env
            else:
                chosen_method = (selection_map.get(config_name, {}).get('method') or 'histogram')
                chosen_T = selection_map.get(config_name, {}).get('T')
                chosen_s = selection_map.get(config_name, {}).get('s')
        method_used = None

        for forecast_step in range(T_TEST_BACKTEST_CFG):
             # For forecasting, Y_current is unknown, so we don't update H_prev based on observation.
             # The func is called with Y_train=None (or an empty slice) as it's not used in forecast mode.
             # We need to pass *some* Y_train for the function signature, but its content won't matter for the forecast calculation itself.
             # Let's pass the last week of the main training data, though it's not used in the forecast formula.

             # Create a dummy Y_train slice for the function call - its content is not used in forecast
             dummy_Y_train_for_forecast = Y_t.iloc[latest_week_index : latest_week_index + 1] # Just needs correct columns


             P_forecast_step, H_t_new_forecast = func(
                 Y_train=dummy_Y_train_for_forecast, # Not used in forecast calculation
                 Y_current=np.zeros(num_targets), # Assume 0 observations for forecasting, Y_Current is already numpy array
                 H_t_prev=current_H_forecast,
                 decay=optimal_decay,
                 jump=optimal_jump
             )
             current_H_forecast = H_t_new_forecast # Update H for the *next* forecast step

             # Apply calibration if requested
             if apply_calibration and chosen_method:
                 if chosen_method in ('histogram', 'isotonic'):
                     if calib_centers is None or calib_rates is None:
                         # bins not available, skip
                         method_used = None
                     else:
                         rates_to_use = calib_rates
                         if cal_apply_smoothing and calib_counts is not None:
                             rates_to_use = _apply_laplace_smoothing(rates_to_use, calib_counts, alpha=cal_laplace_alpha)
                         if chosen_method == 'isotonic':
                             try:
                                 x_fit, y_fit = _fit_isotonic_mapping(calib_centers, rates_to_use, calib_counts if calib_counts is not None else None)
                                 P_forecast_step = _apply_isotonic(P_forecast_step, x_fit, y_fit)
                                 method_used = 'isotonic'
                             except Exception:
                                 P_forecast_step = _apply_histogram_calibration(P_forecast_step, calib_centers, rates_to_use)
                                 method_used = 'histogram_binning'
                         else:
                             P_forecast_step = _apply_histogram_calibration(P_forecast_step, calib_centers, rates_to_use)
                             method_used = 'histogram_binning'
                 elif chosen_method == 'temperature':
                     # p' = sigmoid(logit(p)/T)
                     eps = 1e-12
                     p = np.clip(P_forecast_step, eps, 1 - eps)
                     logits = np.log(p) - np.log(1 - p)
                     T = float(chosen_T) if chosen_T is not None else 1.0
                     P_forecast_step = 1.0 / (1.0 + np.exp(-(logits / max(T, eps))))
                     method_used = 'temperature'
                 elif chosen_method == 'intensity':
                     # λ' = s·λ, p' = 1 - exp(-λ')
                     eps = 1e-12
                     lam = -np.log(np.clip(1 - P_forecast_step, eps, 1.0))
                     s_val = float(chosen_s) if chosen_s is not None else 1.0
                     lam_s = s_val * lam
                     P_forecast_step = 1.0 - np.exp(-lam_s)
                     method_used = 'intensity'

             forecast_probabilities.append(P_forecast_step)


        # Calculate the average weekly probability over the forecast horizon for each target
        average_weekly_prob_per_target = np.mean(forecast_probabilities, axis=0)


        # Store forecast results, including average weekly probabilities
        forecast_data = {
            'model_config': config_name,
            'forecast_date': pd.Timestamp.now(), # Timestamp when forecast was generated
            'forecast_start_week': forecast_start_date,
            'forecast_horizon_weeks': T_TEST_BACKTEST_CFG,
            'forecast_probabilities': np.array(forecast_probabilities), # Store as numpy array
            'average_weekly_probabilities': average_weekly_prob_per_target, # Store average probabilities
            'target_labels': Y_t.columns.tolist(),
            'optimal_decay': optimal_decay, # Store the optimal decay parameter
            'optimal_jump': optimal_jump,   # Store the optimal jump parameter
            'calibration_applied': bool(apply_calibration and chosen_method),
            'calibration_method': (method_used if (apply_calibration and chosen_method) else None),
            'calibration_source': (calib_path if method_used in ('histogram_binning','isotonic') else (comparison_path if method_used in ('temperature','intensity') else None)),
            'calibration_param': (float(chosen_T) if method_used == 'temperature' and (chosen_T is not None) else (float(chosen_s) if method_used == 'intensity' and (chosen_s is not None) else None)),
            'calibration_smoothing_alpha': (cal_laplace_alpha if (apply_calibration and cal_apply_smoothing and method_used in ('histogram_binning','isotonic')) else None)
        }
        all_forecast_results.append(forecast_data)

    print("\n--- Daily Forecast Process Completed ---")
    return all_forecast_results

"""### 6. Reporting and Output

This section contains functions to generate the risk report in different formats (Markdown or JSON) and includes utilities for assigning risk bands and providing interpretations.
"""

# Assume calculate_log_loss, load_latest_data, prepare_data,
# baseline_forecast, contagion_only_forecast, hybrid_forecast,
# run_daily_forecast, model_forecast_configs, MIN_AVG_WEEKLY_PROBABILITY,
# and PROBABILITY_BANDS are already defined in previous cells.

import json # Ensure json is imported here for JSON output
from IPython.display import Markdown, display # Ensure display utilities are imported here
import pandas as pd # Ensure pandas is imported here


# Define probability bands and their thresholds (Average Weekly Probability) - Redefine here or ensure global access
# PROBABILITY_BANDS = { ... } # Assuming this is globally accessible or passed


# Define MIN_AVG_WEEKLY_PROBABILITY - Redefine here or ensure global access
# MIN_AVG_WEEKLY_PROBABILITY = 0.005 # Assuming this is globally accessible or passed


def dataframe_to_markdown(df, columns=None):
    """
    Convert a DataFrame to markdown table format without requiring tabulate dependency.
    
    Args:
        df: pandas DataFrame
        columns: list of column names to include (if None, uses all columns)
    
    Returns:
        str: Markdown formatted table
    """
    if columns:
        df = df[columns]
    
    # Get column names
    cols = df.columns.tolist()
    
    # Start building the markdown table
    markdown = "| " + " | ".join(cols) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    
    # Add data rows
    for _, row in df.iterrows():
        row_values = [str(row[col]) for col in cols]
        markdown += "| " + " | ".join(row_values) + " |\n"
    
    return markdown


def assign_risk_band(probability):
    """Assigns a risk band based on average weekly probability."""
    # Sort bands by threshold in descending order to ensure correct assignment
    sorted_bands = sorted(PROBABILITY_BANDS.items(), key=lambda item: item[1], reverse=True)
    for band, threshold in sorted_bands:
        if probability >= threshold:
            return band
    return 'Very Low' # Should not be reached if 0.0 is in PROBABILITY_BANDS

def get_risk_band_interpretation(band, horizon_weeks):
    """Provides actionable interpretation for each risk band."""
    # Access PROBABILITY_BANDS from global scope or pass it
    interpretations = {
        'Very High': f"**Very High Risk:** Targets in this band have an average weekly probability of {PROBABILITY_BANDS['Very High']:.2%} or higher over the next {horizon_weeks} weeks. **Action:** Immediate review and heightened monitoring recommended. These are the most critical targets to prioritize.",
        'High': f"**High Risk:** Targets with an average weekly probability between {PROBABILITY_BANDS['High']:.2%} and {PROBABILITY_BANDS['Very High']:.2%} over the next {horizon_weeks} weeks. **Action:** Proactive assessment and targeted defense measures should be considered.",
        'Medium': f"**Medium Risk:** Targets with an average weekly probability between {PROBABILITY_BANDS['Medium']:.2%} and {PROBABILITY_BANDS['High']:.2%} over the next {horizon_weeks} weeks. **Action:** Regular monitoring and standard security practices are essential.",
        'Low': f"**Low Risk:** Targets with an average weekly probability between {PROBABILITY_BANDS['Low']:.2%} and {PROBABILITY_BANDS['Medium']:.2%} over the next {horizon_weeks} weeks. **Action:** Standard monitoring and baseline security measures are likely sufficient.",
        'Very Low': f"**Very Low Risk:** Targets with an average weekly probability below {PROBABILITY_BANDS['Low']:.2%} over the next {horizon_weeks} weeks. **Action:** These targets currently represent minimal predicted risk based on the model.",
    }
    return interpretations.get(band, "No interpretation available for this band.")


def generate_risk_report(forecast_results, min_avg_weekly_probability, probability_bands, output_format='markdown', output_file=None, include_calibration_sections: bool = True):
    """
    Generates a comprehensive risk report in Markdown or JSON format.

    Args:
        forecast_results (list): List of dictionaries from run_daily_forecast.
        min_avg_weekly_probability (float): Threshold for filtering targets.
        probability_bands (dict): Dictionary defining risk bands and thresholds.
        output_format (str): 'markdown' or 'json'.
        output_file (str, optional): Path to save the report file.

    Returns:
        str: The report content if output_format is 'markdown' and output_file is None.
        dict: The report content as a dictionary if output_format is 'json'.
        None: If output_file is specified.
    """
    report_content = {} # Dictionary to hold report data for JSON and structured access

    report_content['report_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    report_content['min_avg_weekly_probability_threshold'] = min_avg_weekly_probability
    report_content['probability_bands'] = probability_bands

    report_content['model_forecasts'] = []

    # Try to load calibration summary for inclusion
    calibration_dir = _calibration_assets_dir(os.path.dirname(os.path.abspath(__file__)))
    calibration_summary_path = os.path.join(calibration_dir, 'calibration_summary.json')
    if not os.path.isfile(calibration_summary_path):
        # fallback to legacy path
        legacy_dir = _legacy_calibration_dir(os.path.dirname(os.path.abspath(__file__)))
        legacy_path = os.path.join(legacy_dir, 'calibration_summary.json')
        if os.path.isfile(legacy_path):
            calibration_dir = legacy_dir
            calibration_summary_path = legacy_path
    calibration_summary = None
    if os.path.isfile(calibration_summary_path):
        try:
            with open(calibration_summary_path, 'r', encoding='utf-8') as f:
                calibration_summary = json.load(f)
        except Exception:
            calibration_summary = None

    # Try load segment breakdown for concise per-segment snapshot
    segment_breakdown_path = os.path.join(calibration_dir, 'segment_breakdown.csv')
    segment_df = None
    if os.path.isfile(segment_breakdown_path):
        try:
            segment_df = pd.read_csv(segment_breakdown_path)
        except Exception:
            segment_df = None

    # Rounding helpers
    def _round_float(val, ndigits):
        try:
            return round(float(val), ndigits)
        except Exception:
            return val

    def _round_dict_values(d, ndigits):
        return {k: _round_float(v, ndigits) for k, v in d.items()}

    # Helper to convert absolute paths to relative (to this script directory)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    def _to_relpath(p):
        try:
            if not p:
                return p
            rel = os.path.relpath(p, start=_script_dir)
            # Normalize to forward slashes for Markdown/URLs so they render cross-platform
            return rel.replace(os.sep, '/')
        except Exception:
            return p

    for forecast in forecast_results:
        model_config_name = forecast['model_config']
        forecast_date = forecast['forecast_date'].strftime('%Y-%m-%d %H:%M:%S')
        forecast_start_week = forecast['forecast_start_week'].strftime('%Y-%m-%d')
        forecast_horizon_weeks = forecast['forecast_horizon_weeks']
        average_weekly_probabilities = forecast['average_weekly_probabilities']
        target_labels = forecast['target_labels']

        model_data = {
            'model_config': model_config_name,
            'forecast_date': forecast_date,
            'forecast_start_week': forecast_start_week,
            'forecast_horizon_weeks': forecast_horizon_weeks,
            # Round optimal parameters for readability in JSON
            'optimal_decay': _round_float(forecast.get('optimal_decay'), 3) if forecast.get('optimal_decay') is not None else None,
            'optimal_jump': _round_float(forecast.get('optimal_jump'), 3) if forecast.get('optimal_jump') is not None else None,
            'calibration_applied': forecast.get('calibration_applied', False),
            'calibration_method': forecast.get('calibration_method'),
            # Store calibration source as a relative path for portability/readability
            'calibration_source': _to_relpath(forecast.get('calibration_source')),
            'calibration_param': forecast.get('calibration_param'),
            'calibration_smoothing_alpha': forecast.get('calibration_smoothing_alpha'),
            'average_weekly_probabilities_per_target': {}, # Store probabilities as dict for JSON
            'filtered_targets': [],
            'combined_risk_bundesland': [],
            'combined_risk_sector': [],
            'risk_band_summary': []
        }

        target_probabilities = pd.Series(average_weekly_probabilities, index=target_labels)
        # Store rounded probabilities in JSON for readability (retain numeric type)
        model_data['average_weekly_probabilities_per_target'] = _round_dict_values(target_probabilities.to_dict(), 4)

        # Apply filtering threshold
        filtered_targets_probabilities = target_probabilities[target_probabilities >= min_avg_weekly_probability]

        if not filtered_targets_probabilities.empty:
            # Create DataFrame for filtered targets
            filtered_risk_df = pd.DataFrame({
                'Target': filtered_targets_probabilities.index,
                'Average Weekly Probability': filtered_targets_probabilities.values
            })

            # Extract Bundesland and Sector
            filtered_risk_df['Bundesland'] = filtered_risk_df['Target'].apply(lambda x: x.split(' | ')[0])
            filtered_risk_df['Sector'] = filtered_risk_df['Target'].apply(lambda x: x.split(' | ')[1])

            # Assign Risk Band
            filtered_risk_df['Risk Band'] = filtered_risk_df['Average Weekly Probability'].apply(assign_risk_band)

            # --- Sorting Filtered Targets by Average Weekly Probability (Descending) ---
            # Sort by numerical probability first
            filtered_risk_df = filtered_risk_df.sort_values('Average Weekly Probability', ascending=False)

            # Round probability for JSON output while keeping numeric type
            filtered_risk_df['Average Weekly Probability'] = filtered_risk_df['Average Weekly Probability'].apply(lambda x: _round_float(x, 4))
            model_data['filtered_targets'] = filtered_risk_df[['Target', 'Average Weekly Probability', 'Risk Band']].to_dict(orient='records')


            # Calculate Combined Cumulative Risk per Bundesland
            combined_risk_per_bundesland_filtered = filtered_risk_df.groupby('Bundesland')['Average Weekly Probability'].sum().reset_index()
            combined_risk_per_bundesland_filtered.rename(columns={'Average Weekly Probability': 'Combined Cumulative Risk (Approx)'}, inplace=True)

            # Calculate Combined Cumulative Risk per Sector
            combined_risk_per_sector_filtered = filtered_risk_df.groupby('Sector')['Average Weekly Probability'].sum().reset_index()
            combined_risk_per_sector_filtered.rename(columns={'Average Weekly Probability': 'Combined Cumulative Risk (Approx)'}, inplace=True)

            # Normalize combined risk values for scaling
            max_bundesland_risk = combined_risk_per_bundesland_filtered['Combined Cumulative Risk (Approx)'].max()
            if max_bundesland_risk > 0:
                 combined_risk_per_bundesland_filtered['Scaled Risk (%)'] = (combined_risk_per_bundesland_filtered['Combined Cumulative Risk (Approx)'] / max_bundesland_risk) * 100
            else:
                 combined_risk_per_bundesland_filtered['Scaled Risk (%)'] = 0.0

            max_sector_risk = combined_risk_per_sector_filtered['Combined Cumulative Risk (Approx)'].max()
            if max_sector_risk > 0:
                 combined_risk_per_sector_filtered['Scaled Risk (%)'] = (combined_risk_per_sector_filtered['Combined Cumulative Risk (Approx)'] / max_sector_risk) * 100
            else:
                 combined_risk_per_sector_filtered['Scaled Risk (%)'] = 0.0

            # --- Sorting Combined Risk by Scaled Risk (%) (Descending) ---
            combined_risk_per_bundesland_filtered = combined_risk_per_bundesland_filtered.sort_values('Scaled Risk (%)', ascending=False)
            combined_risk_per_sector_filtered = combined_risk_per_sector_filtered.sort_values('Scaled Risk (%)', ascending=False)


            # Round combined risk values for JSON readability
            combined_risk_per_bundesland_filtered['Combined Cumulative Risk (Approx)'] = combined_risk_per_bundesland_filtered['Combined Cumulative Risk (Approx)'].apply(lambda x: _round_float(x, 3))
            combined_risk_per_bundesland_filtered['Scaled Risk (%)'] = combined_risk_per_bundesland_filtered['Scaled Risk (%)'].apply(lambda x: _round_float(x, 2))
            combined_risk_per_sector_filtered['Combined Cumulative Risk (Approx)'] = combined_risk_per_sector_filtered['Combined Cumulative Risk (Approx)'].apply(lambda x: _round_float(x, 3))
            combined_risk_per_sector_filtered['Scaled Risk (%)'] = combined_risk_per_sector_filtered['Scaled Risk (%)'].apply(lambda x: _round_float(x, 2))

            # Store combined risk data for JSON
            model_data['combined_risk_bundesland'] = combined_risk_per_bundesland_filtered.to_dict(orient='records')
            model_data['combined_risk_sector'] = combined_risk_per_sector_filtered.to_dict(orient='records')

            # Calculate Summary by Risk Band
            risk_band_summary = filtered_risk_df['Risk Band'].value_counts().reset_index()
            risk_band_summary.columns = ['Risk Band', 'Number of Targets']
            # Sort by the defined order in PROBABILITY_BANDS (reversed for descending risk)
            band_order_summary = list(probability_bands.keys())[::-1] # Use passed probability_bands
            risk_band_summary['Risk Band'] = pd.Categorical(risk_band_summary['Risk Band'], categories=band_order_summary, ordered=True)
            risk_band_summary = risk_band_summary.sort_values('Risk Band')


            # Store risk band summary for JSON
            model_data['risk_band_summary'] = risk_band_summary.to_dict(orient='records')


        # Attach calibration metrics if available in summary (only when including calibration)
        if include_calibration_sections and calibration_summary:
            # Find matching record by model_config name (exact match expected)
            matches = [r for r in calibration_summary if r.get('model_config') == model_config_name]
            if matches:
                model_data['calibration_metrics'] = {
                    'ece': _round_float(matches[0].get('ece'), 4),
                    'brier': _round_float(matches[0].get('brier'), 4),
                    'nll': _round_float(matches[0].get('nll'), 6) if matches[0].get('nll') is not None else None,
                    'test_window_weeks': matches[0].get('test_window_weeks'),
                    # Use relative path for the summary source reference
                    'summary_source': _to_relpath(calibration_summary_path)
                }

        report_content['model_forecasts'].append(model_data)


    # Attach a compact calibration summary at the root for programmatic access (only when including calibration)
    if include_calibration_sections and calibration_summary:
        try:
            cal_dir = calibration_dir
            # Determine generation timestamp from calibration_summary.json mtime
            try:
                mtime = os.path.getmtime(calibration_summary_path)
                cal_generated_at = pd.to_datetime(mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                cal_generated_at = None
            report_content['calibration'] = {
                # Store artifacts dir as relative path for readability
                'artifacts_dir': _to_relpath(cal_dir),
                'generated_at': cal_generated_at,
                'summary': [
                    {
                        'model_config': r.get('model_config'),
                        'test_window_weeks': r.get('test_window_weeks'),
                        'ece': _round_float(r.get('ece'), 4),
                        'brier': _round_float(r.get('brier'), 4),
                        'nll': _round_float(r.get('nll'), 6) if r.get('nll') is not None else None
                    } for r in calibration_summary
                ]
            }
        except Exception:
            pass

    if output_format == 'json':
        # Custom encoder to handle numpy types if necessary (though to_dict should handle most)
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        if output_file:
            # Ensure parent directory exists
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report_content, f, indent=4, cls=NpEncoder)
            print(f"Report saved as JSON file: {output_file}")
            return None
        else:
            # Return JSON string if no file specified
            return json.dumps(report_content, indent=4, cls=NpEncoder)


    elif output_format == 'markdown':
        # Base directory for Markdown-relative paths: prefer the output file's folder when provided
        md_base_dir = None
        try:
            md_base_dir = os.path.dirname(output_file) if output_file else _script_dir
        except Exception:
            md_base_dir = _script_dir

        def _to_rel_for_md(p: str | None):
            try:
                if not p:
                    return p
                rel = os.path.relpath(p, start=md_base_dir)
                return rel.replace(os.sep, '/')
            except Exception:
                return p
        markdown_output = f"# ClusterCast Risk Forecast Report ({report_content['report_timestamp']})\n\n"
        markdown_output += f"This report provides a daily forecast of ransomware risk for different Bundesland/Sector combinations using optimized Hybrid and Contagion-Only models.\n\n"
        markdown_output += f"Minimum Average Weekly Probability Threshold for Filtering: {min_avg_weekly_probability:.2%}\n\n"

        # Executive Summary: top risks and key metrics
        try:
            markdown_output += "## Executive Summary\n\n"
            # Top risks across all models (by probability)
            all_filtered = []
            for m in report_content['model_forecasts']:
                for rec in (m.get('filtered_targets') or []):
                    all_filtered.append({
                        'Target': rec['Target'],
                        'Average Weekly Probability': float(rec['Average Weekly Probability']),
                        'Model': m['model_config']
                    })
            if all_filtered:
                top_df = pd.DataFrame(all_filtered)
                top_df = top_df.sort_values('Average Weekly Probability', ascending=False).head(5)
                # Format for display
                top_df_display = top_df.copy()
                top_df_display['Average Weekly Probability'] = top_df_display['Average Weekly Probability'].apply(lambda x: f"{x:.2%}")
                markdown_output += "### Top Risks (All Models)\n\n"
                markdown_output += dataframe_to_markdown(top_df_display[['Target','Average Weekly Probability','Model']]) + "\n\n"
            # Calibration key metrics (exclude when splitting)
            if include_calibration_sections and calibration_summary:
                cal_df = pd.DataFrame(calibration_summary)
                cal_df_small = cal_df[['model_config','test_window_weeks','ece','brier','nll']].copy() if 'nll' in cal_df.columns else cal_df[['model_config','test_window_weeks','ece','brier']].copy()
                for c in ['ece','brier'] + (['nll'] if 'nll' in cal_df_small.columns else []):
                    cal_df_small[c] = cal_df_small[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                markdown_output += "### Calibration Snapshot (ECE/Brier/NLL)\n\n"
                rename_map = {'model_config':'Model','test_window_weeks':'Horizon (weeks)','ece':'ECE','brier':'Brier'}
                if 'nll' in cal_df_small.columns:
                    rename_map['nll'] = 'NLL'
                markdown_output += dataframe_to_markdown(cal_df_small.rename(columns=rename_map)) + "\n\n"
        except Exception:
            pass

        # Global calibration notes if any (exclude when splitting)
        if include_calibration_sections and calibration_summary:
            markdown_output += "## Calibration Summary\n"
            markdown_output += "Calibrated models use histogram binning derived from recent backtests to reduce miscalibration. Metrics below are out-of-sample.\n\n"
            # Build a compact table
            try:
                cal_df = pd.DataFrame(calibration_summary)
                cols = ['model_config', 'test_window_weeks', 'ece', 'brier'] + (['nll'] if 'nll' in cal_df.columns else [])
                cal_display = cal_df[cols].copy()
                rename_cols = {'model_config': 'Model', 'test_window_weeks': 'Horizon (weeks)', 'ece': 'ECE', 'brier': 'Brier'}
                if 'nll' in cal_display.columns:
                    rename_cols['nll'] = 'NLL'
                cal_display.rename(columns=rename_cols, inplace=True)
                # Format numeric columns
                num_cols = ['ECE', 'Brier'] + (['NLL'] if 'NLL' in cal_display.columns else [])
                for col in num_cols:
                    cal_display[col] = cal_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                markdown_output += dataframe_to_markdown(cal_display) + "\n\n"
                # Add note for plots directory (relative path)
                markdown_output += f"Reliability plots are available under: {_to_rel_for_md(calibration_dir)}\n\n"
            except Exception:
                pass

        # Add calibration plots with calibrated overlays (if available) (exclude when splitting)
        try:
            if include_calibration_sections and calibration_summary:
                markdown_output += "## Calibration Plots (raw + calibrated overlay)\n\n"
                # For each model in summary, try to link to reliability_<slug>.png
                for r in calibration_summary:
                    name = r.get('model_config')
                    if not name:
                        continue
                    slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
                    plot_path = os.path.join(calibration_dir, f"reliability_{slug}.png")
                    plot_rel = _to_rel_for_md(plot_path)
                    if os.path.isfile(plot_path):
                        markdown_output += f"### {name}\n\n"
                        markdown_output += f"![Reliability {name}]({plot_rel})\n\n"
                    # Probability histogram
                    hist_path = os.path.join(calibration_dir, f"histogram_{slug}.png")
                    hist_rel = _to_rel_for_md(hist_path)
                    if os.path.isfile(hist_path):
                        markdown_output += f"![Probability Histogram {name}]({hist_rel})\n\n"
        except Exception:
            pass

        # Add reliability-bin CI tables (exclude when splitting)
        try:
            if include_calibration_sections and calibration_summary:
                markdown_output += "## Reliability Bin Confidence Intervals\n\n"
                for r in calibration_summary:
                    name = r.get('model_config')
                    if not name:
                        continue
                    slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
                    bins_csv = os.path.join(calibration_dir, f"bins_{slug}.csv")
                    if os.path.isfile(bins_csv):
                        try:
                            bins_df = pd.read_csv(bins_csv)
                            # Render compact subset of columns
                            cols = [
                                'bin_center',
                                'bin_count',
                                'mean_pred',
                                'event_rate',
                            ]
                            # Optionally include CI columns if present
                            if {'event_rate_lo95','event_rate_hi95'}.issubset(set(bins_df.columns)):
                                cols += ['event_rate_lo95','event_rate_hi95']
                            bins_disp = bins_df[cols].copy()
                            # Formatting
                            for c in ['mean_pred','event_rate']:
                                if c in bins_disp.columns:
                                    bins_disp[c] = bins_disp[c].apply(lambda x: f"{float(x):.4f}")
                            for c in ['event_rate_lo95','event_rate_hi95']:
                                if c in bins_disp.columns:
                                    bins_disp[c] = bins_disp[c].apply(lambda x: f"{float(x):.4f}")
                            bins_disp.rename(columns={
                                'bin_center':'Bin center',
                                'bin_count':'N',
                                'mean_pred':'Mean p',
                                'event_rate':'Obs rate',
                                'event_rate_lo95':'Obs lo95',
                                'event_rate_hi95':'Obs hi95'
                            }, inplace=True)
                            markdown_output += f"### {name}\n\n"
                            markdown_output += dataframe_to_markdown(bins_disp) + "\n\n"
                        except Exception:
                            continue
        except Exception:
            pass

        # Methods used table (Model, Horizon, Calibration Method, Smoothing alpha, Source)
        try:
            if include_calibration_sections and report_content['model_forecasts']:
                methods_rows = []
                for m in report_content['model_forecasts']:
                    param_val = m.get('calibration_param')
                    method = m.get('calibration_method') or ''
                    param_label = ''
                    if method == 'temperature' and param_val is not None:
                        param_label = f" (T={float(param_val):.2f})"
                    elif method == 'intensity' and param_val is not None:
                        param_label = f" (s={float(param_val):.2f})"
                    methods_rows.append({
                        'Model': m.get('model_config'),
                        'Horizon (weeks)': m.get('forecast_horizon_weeks'),
                        'Calibration Method': (method + param_label) if method else '',
                        'Laplace alpha': (f"{m.get('calibration_smoothing_alpha'):.2f}" if m.get('calibration_smoothing_alpha') is not None else ''),
                        'Source': (_to_rel_for_md(os.path.join(_script_dir, m.get('calibration_source'))) if m.get('calibration_source') else '')
                    })
                methods_df = pd.DataFrame(methods_rows)
                markdown_output += "## Methods used\n\n"
                markdown_output += dataframe_to_markdown(methods_df) + "\n\n"
        except Exception:
            pass

        # Delta vs Baseline (NLL, Brier) per Horizon
        try:
            if include_calibration_sections and calibration_summary:
                cal_df = pd.DataFrame(calibration_summary)
                # If NLL missing, compute NLL as None-safe
                if 'nll' not in cal_df.columns:
                    cal_df['nll'] = None
                rows = []
                for horizon, grp in cal_df.groupby('test_window_weeks'):
                    bl = grp[grp['model_config'].str.startswith('baseline')]
                    if bl.empty:
                        continue
                    bl_row = bl.iloc[0]
                    bl_nll = bl_row.get('nll')
                    bl_brier = bl_row.get('brier')
                    for _, r in grp.iterrows():
                        if r['model_config'].startswith('baseline'):
                            continue
                        d_nll = (r['nll'] - bl_nll) if (pd.notnull(r['nll']) and pd.notnull(bl_nll)) else None
                        d_brier = (r['brier'] - bl_brier) if (pd.notnull(r['brier']) and pd.notnull(bl_brier)) else None
                        rows.append({
                            'Model': r['model_config'],
                            'Horizon (weeks)': int(horizon),
                            'ΔNLL': (f"{d_nll:+.6f}" if d_nll is not None else "n/a"),
                            'ΔBrier': (f"{d_brier:+.6f}" if d_brier is not None else "n/a")
                        })
                if rows:
                    delta_df = pd.DataFrame(rows)
                    markdown_output += "## Delta vs Baseline (NLL, Brier)\n\n"
                    markdown_output += "Negative deltas indicate improvement over baseline.\n\n"
                    markdown_output += dataframe_to_markdown(delta_df[['Model','Horizon (weeks)','ΔNLL','ΔBrier']]) + "\n\n"
        except Exception:
            pass

        # Per-Segment Calibration Snapshot (top/bottom ECE)
        try:
            if segment_df is not None and not segment_df.empty:
                seg = segment_df.copy()
                # Average ECE/Brier by segment across all models/horizons
                agg = seg.groupby(['segment_type','segment'], as_index=False).agg(
                    ece_mean=('ece','mean'), brier_mean=('brier','mean'), n=('n_predictions','sum')
                )
                out_lines = []
                for seg_type in ['Bundesland','Sector']:
                    sub = agg[agg['segment_type']==seg_type]
                    if sub.empty:
                        continue
                    best = sub.nsmallest(3, 'ece_mean')
                    worst = sub.nlargest(3, 'ece_mean')
                    out_lines.append(f"### Per-Segment Calibration Snapshot - {seg_type}\n")
                    out_lines.append("Top (lowest ECE):")
                    best_disp = best[['segment','ece_mean','brier_mean','n']].copy()
                    best_disp.rename(columns={'segment':'Segment','ece_mean':'ECE (avg)','brier_mean':'Brier (avg)','n':'N'}, inplace=True)
                    best_disp['ECE (avg)'] = best_disp['ECE (avg)'].apply(lambda x: f"{x:.4f}")
                    best_disp['Brier (avg)'] = best_disp['Brier (avg)'].apply(lambda x: f"{x:.4f}")
                    out_lines.append(dataframe_to_markdown(best_disp))
                    out_lines.append("\nWorst (highest ECE):")
                    worst_disp = worst[['segment','ece_mean','brier_mean','n']].copy()
                    worst_disp.rename(columns={'segment':'Segment','ece_mean':'ECE (avg)','brier_mean':'Brier (avg)','n':'N'}, inplace=True)
                    worst_disp['ECE (avg)'] = worst_disp['ECE (avg)'].apply(lambda x: f"{x:.4f}")
                    worst_disp['Brier (avg)'] = worst_disp['Brier (avg)'].apply(lambda x: f"{x:.4f}")
                    out_lines.append(dataframe_to_markdown(worst_disp))
                    out_lines.append("\n")
                if out_lines:
                    markdown_output += "## Per-Segment Calibration Snapshot (Top/Bottom ECE)\n\n" + "\n".join(out_lines)
        except Exception:
            pass

        markdown_output += "## Risk Band Interpretations\n"
        markdown_output += "Risk bands categorize targets based on their Average Weekly Probability over the forecast horizon. The thresholds and interpretations below are examples and should be adjusted based on operational context.\n\n"
        # Sort bands for consistent display order
        sorted_bands_display = sorted(probability_bands.items(), key=lambda item: item[1], reverse=True)
        for band, threshold in sorted_bands_display:
             markdown_output += f"* **{band} (>= {threshold:.2%}):** {get_risk_band_interpretation(band, 'Forecast Horizon')}\n" # Use placeholder for horizon in general interpretation
        markdown_output += "\n"


        for model_data in report_content['model_forecasts']:
            markdown_output += f"## Model: {model_data['model_config']}\n\n"
            # Compact header line with key details
            parts = []
            # Horizon and start
            parts.append(f"Horizon: {model_data['forecast_horizon_weeks']}w")
            parts.append(f"Start: {model_data['forecast_start_week']}")
            # Optimal parameters
            if model_data.get('optimal_decay') is not None and model_data.get('optimal_jump') is not None:
                parts.append(f"Params: DECAY={model_data['optimal_decay']:.2f}, JUMP={model_data['optimal_jump']:.2f}")
            # Calibration status: always reflect if calibration was applied.
            if model_data.get('calibration_applied'):
                if include_calibration_sections:
                    method = model_data.get('calibration_method') or 'n/a'
                    alpha = model_data.get('calibration_smoothing_alpha')
                    param = model_data.get('calibration_param')
                    # build cal descriptor
                    cal_bits = [method]
                    if method in ('histogram_binning','isotonic') and alpha is not None:
                        cal_bits.append(f"alpha={alpha:.2f}")
                    if method == 'temperature' and param is not None:
                        cal_bits.append(f"T={float(param):.2f}")
                    if method == 'intensity' and param is not None:
                        cal_bits.append(f"s={float(param):.2f}")
                    parts.append("Cal: " + ", ".join(cal_bits))
                    # Metrics: include ECE, Brier, NLL if available
                    if model_data.get('calibration_metrics'):
                        ece = model_data['calibration_metrics'].get('ece')
                        brier = model_data['calibration_metrics'].get('brier')
                        nll = model_data['calibration_metrics'].get('nll')
                        metrics_bits = []
                        if ece is not None and pd.notnull(ece):
                            metrics_bits.append(f"ECE={float(ece):.4f}")
                        if brier is not None and pd.notnull(brier):
                            metrics_bits.append(f"Brier={float(brier):.4f}")
                        if nll is not None and pd.notnull(nll):
                            metrics_bits.append(f"NLL={float(nll):.6f}")
                        if metrics_bits:
                            parts.append("Cal metrics: " + ", ".join(metrics_bits))
                else:
                    # In predictions-only report: concise status without details
                    method = model_data.get('calibration_method') or ''
                    label = f"Calibrated{(' (' + method + ')') if method else ''}"
                    parts.append(label)
            else:
                parts.append("Calibration: not applied")

            markdown_output += " | ".join(parts) + "\n\n"

            if model_data['filtered_targets']:
                markdown_output += f"### Filtered Risk per Bundesland | Sector (Meeting {min_avg_weekly_probability:.2%} Avg Weekly Prob Threshold)\n\n"
                # Create DataFrame from filtered targets for Markdown table display
                # Use the already sorted filtered_risk_df which was sorted by numerical probability
                filtered_df_display = pd.DataFrame(model_data['filtered_targets'])

                # Sort by numerical probability (ensure it's numeric before sorting)
                filtered_df_display['Average Weekly Probability_numeric'] = filtered_df_display['Average Weekly Probability'].astype(float) # Assuming JSON stores raw float
                filtered_df_display = filtered_df_display.sort_values('Average Weekly Probability_numeric', ascending=False)

                # Format the probability column *after* sorting for display
                filtered_df_display['Average Weekly Probability'] = filtered_df_display['Average Weekly Probability_numeric'].apply(lambda x: f"{x:.2%}")

                # Drop the temporary numeric column
                filtered_df_display = filtered_df_display.drop(columns='Average Weekly Probability_numeric')


                markdown_output += dataframe_to_markdown(filtered_df_display, ['Target', 'Average Weekly Probability', 'Risk Band']) + "\n\n"

                markdown_output += f"### Combined Cumulative Risk Overview (Filtered Targets Only)\n\n"

                # Bundesland Risk Table
                bundesland_df_display = pd.DataFrame(model_data['combined_risk_bundesland'])
                # Sort by Scaled Risk (%) for display (already sorted before storing in JSON, but re-sorting here for safety and formatting)
                bundesland_df_display['Scaled Risk Sort'] = bundesland_df_display['Scaled Risk (%)'] # Assuming JSON stores raw float
                bundesland_df_display = bundesland_df_display.sort_values('Scaled Risk Sort', ascending=False)

                # Reformat percentage columns *after* sorting
                bundesland_df_display['Combined Cumulative Risk (Approx)'] = bundesland_df_display['Combined Cumulative Risk (Approx)'].apply(lambda x: f"{float(x):.2%}")
                bundesland_df_display['Scaled Risk (%)'] = bundesland_df_display['Scaled Risk Sort'].apply(lambda x: f"{float(x):.2f}%")

                # Drop the temporary numeric column
                bundesland_df_display = bundesland_df_display.drop(columns='Scaled Risk Sort')


                markdown_output += dataframe_to_markdown(bundesland_df_display) + "\n\n"
                markdown_output += "(Note: 'Combined Cumulative Risk (Approx)' is a sum of average weekly probabilities and can exceed 100%. 'Scaled Risk (%)' normalizes this sum relative to the max Bundesland risk for this model's forecast.)\n\n"


                # Sector Risk Table
                sector_df_display = pd.DataFrame(model_data['combined_risk_sector'])
                # Sort by Scaled Risk (%) for display (already sorted before storing in JSON, but re-sorting here for safety and formatting)
                sector_df_display['Scaled Risk Sort'] = sector_df_display['Scaled Risk (%)'] # Assuming JSON stores raw float
                sector_df_display = sector_df_display.sort_values('Scaled Risk Sort', ascending=False)

                # Reformat percentage columns *after* sorting
                sector_df_display['Combined Cumulative Risk (Approx)'] = sector_df_display['Combined Cumulative Risk (Approx)'].apply(lambda x: f"{float(x):.2%}")
                sector_df_display['Scaled Risk (%)'] = sector_df_display['Scaled Risk Sort'].apply(lambda x: f"{float(x):.2f}%")

                 # Drop the temporary numeric column
                sector_df_display = sector_df_display.drop(columns='Scaled Risk Sort')


                markdown_output += dataframe_to_markdown(sector_df_display) + "\n\n"
                markdown_output += "(Note: 'Combined Cumulative Risk (Approx)' is a sum of average weekly probabilities and can exceed 100%. 'Scaled Risk (%)' normalizes this sum relative to the max Sector risk for this model's forecast.)\n\n"

                # Risk Band Summary Table
                risk_band_summary_df_display = pd.DataFrame(model_data['risk_band_summary'])
                # Sort by the defined order in PROBABILITY_BANDS (reversed for descending risk) for display
                band_order_summary_display = list(probability_bands.keys())[::-1]
                risk_band_summary_df_display['Risk Band'] = pd.Categorical(risk_band_summary_df_display['Risk Band'], categories=band_order_summary_display, ordered=True)
                risk_band_summary_df_display = risk_band_summary_df_display.sort_values('Risk Band')

                markdown_output += dataframe_to_markdown(risk_band_summary_df_display) + "\n\n"


            else:
                markdown_output += f"No targets met the minimum average weekly probability threshold ({min_avg_weekly_probability:.2%}) for this model configuration.\n\n"

        if output_file:
            # Ensure parent directory exists
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_output)
            print(f"Report saved as Markdown file: {output_file}")
            return None
        else:
            return markdown_output # Return markdown string


    else:
        print(f"[ERROR] Invalid output_format: {output_format}. Choose 'markdown' or 'json'.")
        return None

# Define configuration variables for the report
MIN_AVG_WEEKLY_PROBABILITY = 0.005 # Example: Filter targets with less than 0.5% average weekly probability
PROBABILITY_BANDS = {
    'Very High': 0.10, # Example: >= 10% average weekly probability
    'High': 0.05,      # Example: >= 5% and < 10%
    'Medium': 0.02,    # Example: >= 2% and < 5%
    'Low': 0.005       # Example: >= 0.5% and < 2%
    # 'Very Low' is implicitly < 0.5% (or the lowest threshold)
}

print(f"MIN_AVG_WEEKLY_PROBABILITY defined as: {MIN_AVG_WEEKLY_PROBABILITY}")
print(f"PROBABILITY_BANDS defined as: {PROBABILITY_BANDS}")

# --- Calibration rebuild helper ---
def _rebuild_calibration_artifacts_if_needed():
    """Rebuild calibration artifacts before forecasting to avoid using stale bins.

    Controlled by env REBUILD_CALIBRATION (default: true). Uses self_optimizing_backtest.run_calibration_analysis
    in a subprocess to avoid import-time side effects.
    """
    # Skip if this process is the calibration subprocess
    if os.environ.get('CALIBRATION_REBUILDING', '0') == '1':
        return
    rebuild = os.environ.get('REBUILD_CALIBRATION', 'true').lower() != 'false'
    if not rebuild:
        print("Skipping calibration rebuild (REBUILD_CALIBRATION=false)")
        return

    # Resolve data path
    data_path = find_data_file(None)
    if not data_path:
        print("[WARNING] Could not resolve data path for calibration rebuild; continuing without rebuild.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = _calibration_assets_dir(script_dir)
    n_bins = os.environ.get('CALIBRATION_N_BINS', '20')
    min_count = os.environ.get('CAL_MIN_COUNT_PER_BIN', '100')

    # Build Python one-liner for subprocess
    code = (
        "from self_optimizing_backtest import run_calibration_analysis, run_calibration_method_comparison; "
        f"run_calibration_analysis('all', r'{data_path}', n_bins={n_bins}, min_count_per_bin={min_count}, output_dir=r'{output_dir}'); "
        f"run_calibration_method_comparison('all', r'{data_path}', n_bins={n_bins}, laplace_alpha=0.5, min_count_per_bin={min_count}, output_dir=r'{output_dir}')"
    )
    print("Rebuilding calibration artifacts via backtest (this may take a few minutes)...")
    try:
        # Use sys.executable to ensure same interpreter
        env = os.environ.copy()
        env['CALIBRATION_REBUILDING'] = '1'  # prevent nested rebuilds in the subprocess
        result = subprocess.run([sys.executable, '-c', code], cwd=script_dir, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print("[WARNING] Calibration rebuild failed. Output:")
            print(result.stdout)
            print(result.stderr)
        else:
            print("Calibration rebuild complete.")
    except Exception as e:
        print(f"[WARNING] Exception during calibration rebuild: {e}")

# --- Run Daily Forecast and Generate Report ---

# The following execution block is guarded to allow safe importing of helpers from this module.

def _backup_existing_markdown(md_path: str):
    try:
        if os.path.isfile(md_path):
            base, ext = os.path.splitext(md_path)
            ts = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
            backup_path = f"{base}.manual.{ts}{ext}"
            shutil.copy2(md_path, backup_path)
            print(f"Backed up existing Markdown to: {backup_path}")
    except Exception as e:
        print(f"[WARNING] Failed to back up existing Markdown: {e}")


def _load_calibration_summary_for_report(script_dir_local: str):
    cal_dir = _calibration_assets_dir(script_dir_local)
    cal_sum = os.path.join(cal_dir, 'calibration_summary.json')
    if not os.path.isfile(cal_sum):
        # fallback legacy
        cal_dir = _legacy_calibration_dir(script_dir_local)
        cal_sum = os.path.join(cal_dir, 'calibration_summary.json')
    if os.path.isfile(cal_sum):
        try:
            with open(cal_sum, 'r', encoding='utf-8') as f:
                return cal_dir, json.load(f)
        except Exception:
            return cal_dir, None
    return cal_dir, None


def generate_calibration_report(output_format: str = 'markdown', output_file: str | None = None):
    """Generate a stand-alone calibration report (md/json) using artifacts in the calibration assets dir.

    The report summarizes calibration metrics, embeds reliability/histogram plots, and renders bin CI tables.
    """
    _script = os.path.dirname(os.path.abspath(__file__))
    cal_dir, cal_summary = _load_calibration_summary_for_report(_script)

    content = {
        'report_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'artifacts_dir': cal_dir.replace(os.sep, '/'),
        'summary': cal_summary or []
    }

    if output_format == 'json':
        if output_file:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=4)
            print(f"Calibration report saved as JSON file: {output_file}")
            return None
        else:
            return json.dumps(content, indent=4)

    # Markdown generation
    md_base_dir = os.path.dirname(output_file) if output_file else _script
    def _to_rel(p: str | None):
        try:
            if not p:
                return p
            rel = os.path.relpath(p, start=md_base_dir)
            return rel.replace(os.sep, '/')
        except Exception:
            return p

    md = f"# Calibration Report ({content['report_timestamp']})\n\n"
    md += "This report summarizes calibration diagnostics, metrics, and artifacts.\n\n"
    # Summary table (raw vs calibrated)
    if cal_summary:
        try:
            cal_df = pd.DataFrame(cal_summary)
            # Support both legacy fields (ece/brier/nll) and new explicit raw/cal fields
            # Build a display with raw, cal, and deltas if available
            def _fmt(x, n=4):
                try:
                    return f"{float(x):.{n}f}"
                except Exception:
                    return ''
            # Ensure expected columns exist
            if {'ece_cal','brier_cal','nll_cal'}.intersection(set(cal_df.columns)):
                # Use explicit fields
                disp = pd.DataFrame({
                    'Model': cal_df['model_config'],
                    'Horizon (weeks)': cal_df['test_window_weeks'],
                    'ECE (raw)': cal_df.get('ece_raw', cal_df.get('ece')),
                    'ECE (cal)': cal_df.get('ece_cal'),
                    'ΔECE': cal_df.get('ece_cal') - cal_df.get('ece_raw', cal_df.get('ece')),
                    'Brier (raw)': cal_df.get('brier_raw', cal_df.get('brier')),
                    'Brier (cal)': cal_df.get('brier_cal'),
                    'ΔBrier': cal_df.get('brier_cal') - cal_df.get('brier_raw', cal_df.get('brier')),
                })
                # NLL optional
                if 'nll_cal' in cal_df.columns or 'nll' in cal_df.columns or 'nll_raw' in cal_df.columns:
                    disp['NLL (raw)'] = cal_df.get('nll_raw', cal_df.get('nll'))
                    disp['NLL (cal)'] = cal_df.get('nll_cal')
                    disp['ΔNLL'] = disp['NLL (cal)'] - disp['NLL (raw)']
                # Format
                for c in ['ECE (raw)','ECE (cal)','ΔECE','Brier (raw)','Brier (cal)','ΔBrier','NLL (raw)','NLL (cal)','ΔNLL']:
                    if c in disp.columns:
                        if c.startswith('Δ'):
                            disp[c] = disp[c].apply(lambda x: f"{float(x):+0.4f}" if pd.notnull(x) else '')
                        else:
                            # NLL with higher precision
                            if 'NLL' in c:
                                disp[c] = disp[c].apply(lambda x: _fmt(x, n=6))
                            else:
                                disp[c] = disp[c].apply(lambda x: _fmt(x, n=4))
                md += "## Summary Metrics (Raw vs Calibrated)\n\n" + dataframe_to_markdown(disp) + "\n\n"
            else:
                # Fallback to legacy simple table
                cols = ['model_config','test_window_weeks','ece','brier'] + (['nll'] if 'nll' in cal_df.columns else [])
                disp = cal_df[cols].copy()
                disp.rename(columns={'model_config':'Model','test_window_weeks':'Horizon (weeks)','ece':'ECE','brier':'Brier','nll':'NLL'}, inplace=True)
                for c in [c for c in ['ECE','Brier','NLL'] if c in disp.columns]:
                    disp[c] = disp[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else '')
                md += "## Summary Metrics\n\n" + dataframe_to_markdown(disp) + "\n\n"
        except Exception:
            pass

    # Plots
    if cal_summary:
        md += "## Plots\n\n"
        for r in cal_summary:
            name = r.get('model_config')
            if not name:
                continue
            slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
            rel_path = os.path.join(cal_dir, f"reliability_{slug}.png")
            hist_path = os.path.join(cal_dir, f"histogram_{slug}.png")
            if os.path.isfile(rel_path) or os.path.isfile(hist_path):
                md += f"### {name}\n\n"
            if os.path.isfile(rel_path):
                md += f"![Reliability {name}]({_to_rel(rel_path)})\n\n"
            if os.path.isfile(hist_path):
                md += f"![Probability Histogram {name}]({_to_rel(hist_path)})\n\n"

    # Bin CI tables
    if cal_summary:
        md += "## Reliability Bin Confidence Intervals\n\n"
        for r in cal_summary:
            name = r.get('model_config')
            if not name:
                continue
            slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
            bins_csv = os.path.join(cal_dir, f"bins_{slug}.csv")
            if not os.path.isfile(bins_csv):
                continue
            try:
                bins_df = pd.read_csv(bins_csv)
                cols = ['bin_center','bin_count','mean_pred','event_rate']
                if {'event_rate_lo95','event_rate_hi95'}.issubset(set(bins_df.columns)):
                    cols += ['event_rate_lo95','event_rate_hi95']
                disp = bins_df[cols].copy()
                for c in ['mean_pred','event_rate','event_rate_lo95','event_rate_hi95']:
                    if c in disp.columns:
                        disp[c] = disp[c].apply(lambda x: f"{float(x):.4f}")
                disp.rename(columns={'bin_center':'Bin center','bin_count':'N','mean_pred':'Mean p','event_rate':'Obs rate','event_rate_lo95':'Obs lo95','event_rate_hi95':'Obs hi95'}, inplace=True)
                md += f"### {name}\n\n" + dataframe_to_markdown(disp) + "\n\n"
            except Exception:
                continue

    if output_file:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"Calibration report saved as Markdown file: {output_file}")
        return None
    return md


if __name__ == "__main__":
    # 1. Run the daily forecast process (pass None to auto-resolve via environment or common locations)
    # EXEC --------------------------------------------------------------------------------
    # Ensure calibration artifacts are freshly rebuilt before generating forecasts
    _rebuild_calibration_artifacts_if_needed()
    daily_forecasts = run_daily_forecast(None, model_forecast_configs)
    # REPORT ------------------------------------------------------------------------------

    # 2. Generate and save the report (Both Markdown and JSON by default)
    # Can be disabled by setting EXPORT_MD=false or EXPORT_JSON=false
    export_md = os.environ.get('EXPORT_MD', 'true').lower() != 'false'
    export_json = os.environ.get('EXPORT_JSON', 'true').lower() != 'false'

    # Resolve export path relative to this script directory when a relative path is provided
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(_calibration_assets_dir(script_dir), exist_ok=True)  # ensure new assets dir exists
    report_base_name = os.environ.get('REPORT_BASE_NAME', 'clustercast_report')

    # Feature flag to split outputs into predictions vs calibration reports (default: true)
    split_outputs = os.environ.get('SPLIT_OUTPUTS', 'true').lower() == 'true'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    def _write_predictions_outputs(predictions_base: str):
        paths = {}
        if export_json:
            json_file = f'{predictions_base}.json'
            json_path = json_file if os.path.isabs(json_file) else os.path.join(output_dir, json_file)
            generate_risk_report(daily_forecasts or [], MIN_AVG_WEEKLY_PROBABILITY, PROBABILITY_BANDS,
                                 output_format='json', output_file=json_path, include_calibration_sections=False)
            paths['json'] = json_path
        if export_md:
            md_file = f'{predictions_base}.md'
            md_path = md_file if os.path.isabs(md_file) else os.path.join(output_dir, md_file)
            _backup_existing_markdown(md_path)
            generate_risk_report(daily_forecasts or [], MIN_AVG_WEEKLY_PROBABILITY, PROBABILITY_BANDS,
                                 output_format='markdown', output_file=md_path, include_calibration_sections=False)
            paths['md'] = md_path
        return paths

    # Generate outputs based on split flag
    if split_outputs:
        # Predictions-only outputs
        preds_paths = _write_predictions_outputs('predictions')
        # Calibration-only outputs
        if export_json:
            cal_json_path = os.path.join(output_dir, 'calibration.json')
            generate_calibration_report(output_format='json', output_file=cal_json_path)
        if export_md:
            cal_md_path = os.path.join(output_dir, 'calibration.md')
            _backup_existing_markdown(cal_md_path)
            generate_calibration_report(output_format='markdown', output_file=cal_md_path)
        # Do not emit legacy combined files in split mode (ensures top-level has only 4 files)
        json_file_path = preds_paths.get('json') if export_json else None
        md_file_path = preds_paths.get('md') if export_md else None
    else:
        # Legacy combined outputs for backward compatibility
        if export_json:
            json_file = os.environ.get('REPORT_JSON_FILE', f'{report_base_name}.json')
            json_file_path = json_file if os.path.isabs(json_file) else os.path.join(output_dir, json_file)
            generate_risk_report(daily_forecasts or [], MIN_AVG_WEEKLY_PROBABILITY, PROBABILITY_BANDS,
                                 output_format='json', output_file=json_file_path, include_calibration_sections=True)

        if export_md:
            md_file = os.environ.get('REPORT_MD_FILE', f'{report_base_name}.md')
            md_file_path = md_file if os.path.isabs(md_file) else os.path.join(output_dir, md_file)
            _backup_existing_markdown(md_file_path)
            generate_risk_report(daily_forecasts or [], MIN_AVG_WEEKLY_PROBABILITY, PROBABILITY_BANDS,
                                 output_format='markdown', output_file=md_file_path, include_calibration_sections=True)

    # 3. Display a brief overview or the report content based on exported formats
    if daily_forecasts:
        print(f"\n--- Report Overview ---")
        if export_json:
            print(f"JSON report saved to: {json_file_path}")
        if export_md:
            print(f"Markdown report saved to: {md_file_path}")
        print(f"Report Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Minimum Average Weekly Probability Threshold for Filtering: {MIN_AVG_WEEKLY_PROBABILITY:.2%}")
        print("\nSummary of Forecasts per Model:")
        for forecast in daily_forecasts:
            model_config_name = forecast['model_config']
            forecast_horizon_weeks = forecast['forecast_horizon_weeks']
            average_weekly_probabilities = forecast['average_weekly_probabilities']
            target_labels = forecast['target_labels']
            optimal_decay = forecast.get('optimal_decay', 'N/A')
            optimal_jump = forecast.get('optimal_jump', 'N/A')
            target_probabilities = pd.Series(average_weekly_probabilities, index=target_labels)
            filtered_targets_probabilities = target_probabilities[target_probabilities >= MIN_AVG_WEEKLY_PROBABILITY]

            print(f"- {model_config_name} (Forecast Horizon: {forecast_horizon_weeks} weeks):")
            
            # Display optimal parameters if available
            if optimal_decay != 'N/A' and optimal_jump != 'N/A':
                print(f"  Optimal Parameters: DECAY={optimal_decay:.2f}, JUMP={optimal_jump:.2f}")
            
            print(f"  Number of targets meeting threshold ({MIN_AVG_WEEKLY_PROBABILITY:.2%}): {len(filtered_targets_probabilities)}")

            if len(filtered_targets_probabilities) > 0:
                # Display top 3 targets in the overview for quick glance
                print("  Top 3 Targets (Average Weekly Probability):")
                top_3_targets = filtered_targets_probabilities.sort_values(ascending=False).head(3)
                for target, prob in top_3_targets.items():
                     print(f"    - {target}: {prob:.2%}")
            else:
                 print("  No targets above threshold.")
    else:
        print("\nNo daily forecasts were generated. An empty report skeleton was still written.")
        if export_json:
            print(f"Empty JSON report saved to: {json_file_path}")
        if export_md:
            print(f"Empty Markdown report saved to: {md_file_path}")