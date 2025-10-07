import pandas as pd
import numpy as np
import os
import json
from typing import Tuple, Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Plotting optional; will handle gracefully if unavailable

# Import necessary functions from the main forecast file
try:
    from forecast import (
        baseline_forecast, 
        contagion_only_forecast, 
        hybrid_forecast, 
        prepare_data, 
        calculate_log_loss,
        DECAY_FIXED,
        JUMP_FIXED
    )
except ImportError:
    # Try importing with underscore instead of hyphen
    try:
        import importlib.util
        import sys
        import os
        
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_file_path = os.path.join(script_dir, 'forecast.py')
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("final_forecast", main_file_path)
        final_forecast = importlib.util.module_from_spec(spec)
        sys.modules["final_forecast"] = final_forecast
        spec.loader.exec_module(final_forecast)
        
        # Import the required functions
        baseline_forecast = final_forecast.baseline_forecast
        contagion_only_forecast = final_forecast.contagion_only_forecast
        hybrid_forecast = final_forecast.hybrid_forecast
        prepare_data = final_forecast.prepare_data
        calculate_log_loss = final_forecast.calculate_log_loss
        DECAY_FIXED = final_forecast.DECAY_FIXED
        JUMP_FIXED = final_forecast.JUMP_FIXED
        
        print("Successfully imported functions from forecast.py")
        
    except Exception as e:
        print(f"[ERROR] Could not import required functions from forecast.py: {e}")
        print("Please ensure forecast.py is in the same directory and contains the required functions.")
        sys.exit(1)

def do_backtest(model_name: str,
                data_file: str = "data/incidents.csv",
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
    :param data_file: The CSV file containing the incident data.
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
            'T_TRAIN_OPT': 4, # Optimal T_TRAIN_OPT for Contagion (8/1)
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
            'T_TRAIN_OPT': 16, # Optimal T_TRAIN_OPT for Hybrid (26/13)
            'T_TEST_OPT': 4    # Optimal T_TEST_OPT for Hybrid (26/13)
        }
        model_configs['hybrid (52/26 Opt)'] = {
            'func': hybrid_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 52,
            'T_TEST_BACKTEST': 26,
            'optimize_params': True,
            'T_TRAIN_OPT': 4, # Optimal T_TRAIN_OPT for Hybrid (52/26)
            'T_TEST_OPT': 1    # Optimal T_TEST_OPT for Hybrid (52/26)
        }


    if not model_configs:
        print(f"[ERROR] Invalid model_name '{model_name}'. Choose 'baseline', 'contagion', 'hybrid', or 'all'.")
        return None

    # Load data once based on the maximum required data length across all configurations
    max_train_backtest = max([config['T_TRAIN_BACKTEST'] for config in model_configs.values()])
    max_test_backtest = max([config['T_TEST_BACKTEST'] for config in model_configs.values()])
    # Determine max optimization window needed across all configs that optimize
    max_train_opt_needed = max([config.get('T_TRAIN_OPT', 0) for config in model_configs.values() if config.get('optimize_params', False)], default=0)
    max_test_opt_needed = max([config.get('T_TEST_OPT', 0) for config in model_configs.values() if config.get('optimize_params', False)], default=0)


    # Ensure enough data for the longest backtest window including the optimization period preceding the very first forecast step
    # The data needs to cover from the earliest point required by any config's first forecast step
    # up to the latest point required by any config's last forecast step.
    # The earliest point is min(actual_start_index for all configs) - max_train_backtest
    # The latest point is max(end_index_cfg for all configs)
    earliest_start_offset = max(max_train_backtest, max_train_opt_needed + max_test_opt_needed) # The minimum number of weeks required before the first forecast step
    latest_end_index = max([config['T_TRAIN_BACKTEST'] + config['T_TEST_BACKTEST'] for config in model_configs.values()])


    try:
        # Prepare data with enough history for all configurations
        Y_t_full, num_targets, total_weeks, Y_t_index = prepare_data(data_file, earliest_start_offset, latest_end_index)

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
                                    Y_obs_counts_mini = Y_t_full.iloc[opt_t_full]
                                    y_true_binary_mini = (Y_obs_counts_mini > 0).astype(int)

                                    P_forecast_mini, H_t_new_mini = func(
                                        Y_train=Y_train_mini,
                                        Y_current=Y_obs_counts_mini,
                                        H_t_prev=current_H_prev_mini,
                                        decay=decay,
                                        jump=jump
                                    )
                                    mini_log_losses.append(calculate_log_loss(y_true_binary_mini.values, P_forecast_mini))
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
        for index, row in results_df[results_df['Model'].str.startswith('baseline')].iterrows():
             baseline_nlls[(row['T_TRAIN_BACKTEST'], row['T_TEST_BACKTEST'])] = row['Average Log Loss (NLL)']

        results_df['Skill Score vs. Baseline (%)'] = results_df.apply(
            lambda row: (baseline_nlls.get((row['T_TRAIN_BACKTEST'], row['T_TEST_BACKTEST']), np.nan) - row['Average Log Loss (NLL)']) / baseline_nlls.get((row['T_TRAIN_BACKTEST'], row['T_TEST_BACKTEST']), np.nan) * 100 if not row['Model'].startswith('baseline') and (row['T_TRAIN_BACKTEST'], row['T_TEST_BACKTEST']) in baseline_nlls else 0.0 if row['Model'].startswith('baseline') else np.nan,
            axis=1
        )

        # Drop the temporary backtest window columns
        results_df = results_df.drop(columns=['T_TRAIN_BACKTEST', 'T_TEST_BACKTEST'])


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
    print(results_df.to_string(index=False))

    return results_df


# =====================
# Calibration utilities
# =====================

def _compute_calibration_stats_adaptive(
    pred: np.ndarray,
    true: np.ndarray,
    n_bins: int = 10,
    min_count_per_bin: int = 100
) -> Dict:
    """
    Compute reliability stats with adaptive (quantile) binning and optional min-count bin merging.

    Args:
        pred: (N,) predicted probabilities in [0,1]
        true: (N,) binary outcomes {0,1}
        n_bins: target number of bins before merging
        min_count_per_bin: minimum samples per bin after merging (best effort)

    Returns:
        dict with keys: bin_edges, bin_centers, bin_counts, bin_mean_pred, bin_event_rate, ece, brier
    """
    pred = np.clip(np.asarray(pred).ravel(), 0.0, 1.0)
    true = np.asarray(true).ravel().astype(float)
    N = len(pred)
    if N == 0:
        return {
            'bin_edges': [], 'bin_centers': [], 'bin_counts': [],
            'bin_mean_pred': [], 'bin_event_rate': [], 'ece': None, 'brier': None
        }

    # Sort by predicted probability
    order = np.argsort(pred)
    p_sorted = pred[order]
    y_sorted = true[order]

    # Initial quantile cut indices
    cuts = [0]
    for k in range(1, n_bins):
        idx = int(round(k * N / n_bins))
        idx = max(cuts[-1], min(idx, N))
        cuts.append(idx)
    cuts.append(N)
    # Build initial bins as (start,end)
    bins = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i] < cuts[i+1]]

    # Merge bins with insufficient count, preferring merge with previous when possible
    def _bin_count(b):
        s, e = b
        return max(0, e - s)

    i = 0
    while i < len(bins):
        if _bin_count(bins[i]) >= min_count_per_bin or len(bins) == 1:
            i += 1
            continue
        # Merge with neighbor
        if i > 0:
            # merge with previous
            s0, e0 = bins[i-1]
            s1, e1 = bins[i]
            bins[i-1] = (s0, e1)
            bins.pop(i)
            i -= 1
        else:
            # i == 0, merge with next
            s0, e0 = bins[i]
            s1, e1 = bins[i+1]
            bins[i] = (s0, e1)
            bins.pop(i+1)
    
    # Compute per-bin stats
    bin_counts = []
    bin_mean_pred = []
    bin_event_rate = []
    bin_low = []
    bin_high = []
    for s, e in bins:
        pp = p_sorted[s:e]
        yy = y_sorted[s:e]
        cnt = int(len(pp))
        bin_counts.append(cnt)
        bin_mean_pred.append(float(np.mean(pp)) if cnt > 0 else 0.0)
        bin_event_rate.append(float(np.mean(yy)) if cnt > 0 else 0.0)
        bin_low.append(float(pp.min()) if cnt > 0 else 0.0)
        bin_high.append(float(pp.max()) if cnt > 0 else 0.0)

    # Define edges as cumulative highs; ensure coverage [0,1]
    edges = [0.0]
    for h in bin_high[:-1]:
        edges.append(float(h))
    edges.append(1.0)
    edges = np.array(edges, dtype=float)

    # ECE with bin weights
    counts_arr = np.array(bin_counts, dtype=float)
    mean_pred_arr = np.array(bin_mean_pred, dtype=float)
    event_rate_arr = np.array(bin_event_rate, dtype=float)
    weights = counts_arr / max(1.0, float(N))
    ece = float(np.sum(weights * np.abs(event_rate_arr - mean_pred_arr))) if weights.size else None

    # Brier
    brier = float(np.mean((pred - true) ** 2))

    # Centers: use mean predicted probability per bin (works well with nearest-center mapping)
    centers = mean_pred_arr.copy()

    return {
        'bin_edges': edges.tolist(),
        'bin_centers': centers.tolist(),
        'bin_counts': counts_arr.astype(int).tolist(),
        'bin_mean_pred': mean_pred_arr.tolist(),
        'bin_event_rate': event_rate_arr.tolist(),
        'ece': ece,
        'brier': brier,
    }


def _apply_laplace_smoothing(event_rates: np.ndarray, bin_counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Laplace smoothing for bin event rates to reduce extreme 0/1 when bins are sparse."""
    event_rates = np.asarray(event_rates, dtype=float)
    bin_counts = np.asarray(bin_counts, dtype=float)
    events = np.clip(event_rates, 0.0, 1.0) * np.clip(bin_counts, 0.0, None)
    smoothed = (events + alpha) / (np.clip(bin_counts, 0.0, None) + 2.0 * alpha)
    return np.clip(smoothed, 0.0, 1.0)


def _fit_isotonic_mapping(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
    """Fit non-decreasing isotonic mapping using PAV over (x,y) with optional weights w."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(w, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]
    y_hat = y.copy()
    weights = w.copy()
    n = len(y_hat)
    i = 0
    while i < n - 1:
        if y_hat[i] <= y_hat[i + 1] + 1e-12:
            i += 1
            continue
        j = i
        while j >= 0 and y_hat[j] > y_hat[j + 1] + 1e-12:
            total_w = weights[j] + weights[j + 1]
            avg = (weights[j] * y_hat[j] + weights[j + 1] * y_hat[j + 1]) / total_w if total_w > 0 else (y_hat[j] + y_hat[j + 1]) / 2.0
            y_hat[j] = avg
            y_hat[j + 1] = avg
            weights[j] = total_w
            if j + 1 < n - 1:
                y_hat = np.delete(y_hat, j + 1)
                x = np.delete(x, j + 1)
                weights = np.delete(weights, j + 1)
                n -= 1
            j -= 1
        i = max(j + 1, 0)
    return x, np.clip(y_hat, 0.0, 1.0)


def _apply_isotonic(probs: np.ndarray, x_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """Apply isotonic mapping via linear interpolation across fitted points."""
    p = np.asarray(probs, dtype=float)
    flat = p.reshape(-1)
    if len(x_fit) < 2:
        return p
    x_fit = np.asarray(x_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    flat_clip = np.clip(flat, x_fit[0], x_fit[-1])
    out = np.interp(flat_clip, x_fit, y_fit)
    return out.reshape(p.shape)


def _wilson_ci(phat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    margin = (z/denom) * np.sqrt((phat*(1 - phat) + (z*z)/(4*n)) / n)
    return max(0.0, center - margin), min(1.0, center + margin)

def _save_reliability_plot(stats_raw: Dict, out_path: str, title: str, stats_cal: Optional[Dict] = None, cal_label: str = 'Calibrated'):
    if plt is None:
        return  # matplotlib not available; skip plotting gracefully
    plt.figure(figsize=(6, 6))
    # Perfect calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Raw curve with Wilson CIs and occupancy annotations
    x = np.array(stats_raw['bin_centers'])
    y = np.array(stats_raw['bin_event_rate'])
    n = np.array(stats_raw['bin_counts'], dtype=int)
    # Error bars per bin
    lowers = []
    uppers = []
    for yi, ni in zip(y, n):
        lo, hi = _wilson_ci(float(yi), int(ni))
        lowers.append(max(0.0, yi - lo))
        uppers.append(max(0.0, hi - yi))
    yerr = np.vstack([lowers, uppers])
    plt.errorbar(x, y, yerr=yerr, fmt='o-', color='#1f77b4', label='Raw (95% CI)')
    # Annotate n
    for xi, yi, ni in zip(x, y, n):
        plt.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='#1f77b4')

    # Calibrated curve if provided
    if stats_cal is not None:
        x2 = np.array(stats_cal['bin_centers'])
        y2 = np.array(stats_cal['bin_event_rate'])
        n2 = np.array(stats_cal['bin_counts'], dtype=int)
        lowers2, uppers2 = [], []
        for yj, nj in zip(y2, n2):
            lo2, hi2 = _wilson_ci(float(yj), int(nj))
            lowers2.append(max(0.0, yj - lo2))
            uppers2.append(max(0.0, hi2 - yj))
        yerr2 = np.vstack([lowers2, uppers2])
        plt.errorbar(x2, y2, yerr=yerr2, fmt='s-', color='#ff7f0e', label=f'{cal_label} (95% CI)')
        # annotate n for calibrated
        for xi, yi, ni in zip(x2, y2, n2):
            plt.annotate(f"n={int(ni)}", (xi, yi), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8, color='#ff7f0e')

    plt.xlabel('Predicted probability')
    plt.ylabel('Observed event rate')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_probability_histogram(pred_raw: np.ndarray, out_path: str, title: str, pred_cal: Optional[np.ndarray] = None):
    if plt is None:
        return
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0.0, 1.0, 31)
    plt.hist(np.clip(pred_raw, 0.0, 1.0), bins=bins, alpha=0.6, color='#1f77b4', label='Raw', edgecolor='none')
    if pred_cal is not None:
        plt.hist(np.clip(pred_cal, 0.0, 1.0), bins=bins, alpha=0.6, color='#ff7f0e', label='Calibrated', edgecolor='none')
    plt.xlim(0, 1)
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_calibration_analysis(
    model_name: str = 'all',
    data_file: str = 'data/incidents.csv',
    n_bins: int = 10,
    min_count_per_bin: int = 100,
    output_dir: str = 'output/calibration'
):
    """
    Run backtests while collecting per-target predictions and outcomes to compute
    calibration statistics (reliability curves, ECE, Brier) and optional plots.

    Artifacts written under output_dir:
      - calibration_summary.json: ECE & Brier per model/window
      - reliability_<slug>.png: Reliability diagrams per model/window (if matplotlib is available)
      - bins_<slug>.csv: Per-bin stats (counts, mean_pred, event_rate)
      - segment_breakdown.csv: Per-segment (Bundesland, Sector) ECE & Brier per model/window
    """
    # Reuse do_backtest logic but collect predictions

    # Define model configurations with optimal windows (aligned with existing do_backtest)
    model_configs = {}
    baseline_configs = {
        'baseline (8/1)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 8, 'T_TEST_BACKTEST': 1, 'optimize_params': False},
        'baseline (26/13)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 26, 'T_TEST_BACKTEST': 13, 'optimize_params': False},
        'baseline (52/26)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 52, 'T_TEST_BACKTEST': 26, 'optimize_params': False},
    }
    if model_name.lower() in ['all', 'baseline']:
        model_configs.update(baseline_configs)
    if model_name.lower() in ['all', 'contagion']:
        model_configs['contagion (8/1 Opt)'] = {
            'func': contagion_only_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 8,
            'T_TEST_BACKTEST': 1,
            'optimize_params': True,
            'T_TRAIN_OPT': 4,
            'T_TEST_OPT': 1
        }
    if model_name.lower() in ['all', 'hybrid']:
        model_configs['hybrid (26/13 Opt)'] = {
            'func': hybrid_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 26,
            'T_TEST_BACKTEST': 13,
            'optimize_params': True,
            'T_TRAIN_OPT': 16,
            'T_TEST_OPT': 4
        }
        model_configs['hybrid (52/26 Opt)'] = {
            'func': hybrid_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 52,
            'T_TEST_BACKTEST': 26,
            'optimize_params': True,
            'T_TRAIN_OPT': 4,
            'T_TEST_OPT': 1
        }

    # Prepare data to cover all configs
    max_train_backtest = max([c['T_TRAIN_BACKTEST'] for c in model_configs.values()])
    max_test_backtest = max([c['T_TEST_BACKTEST'] for c in model_configs.values()])
    max_train_opt = max([c.get('T_TRAIN_OPT', 0) for c in model_configs.values() if c.get('optimize_params', False)], default=0)
    max_test_opt = max([c.get('T_TEST_OPT', 0) for c in model_configs.values() if c.get('optimize_params', False)], default=0)
    earliest_start_offset = max(max_train_backtest, max_train_opt + max_test_opt)
    latest_end_index = max([c['T_TRAIN_BACKTEST'] + c['T_TEST_BACKTEST'] for c in model_configs.values()])

    Y_t_full, num_targets, total_weeks, _ = prepare_data(data_file, earliest_start_offset, latest_end_index)
    target_labels = list(Y_t_full.columns)

    os.makedirs(output_dir, exist_ok=True)
    # Try to load method selection (from prior comparison run)
    selection = {}
    sel_path = os.path.join(output_dir, 'calibration_method_comparison.json')
    if os.path.isfile(sel_path):
        try:
            with open(sel_path, 'r', encoding='utf-8') as f:
                _data = json.load(f)
            for row in (_data.get('per_model') or []):
                name = row.get('model_config')
                if not name:
                    continue
                selection[name] = {
                    'method': (row.get('recommended_method') or '').strip().lower(),
                    'T': (row.get('temperature') or {}).get('T'),
                    's': (row.get('intensity') or {}).get('s')
                }
        except Exception:
            selection = {}
    summary_rows = []
    segment_rows = []

    for config_name, config in model_configs.items():
        func = config['func']
        T_TRAIN_BACKTEST_CFG = config['T_TRAIN_BACKTEST']
        T_TEST_BACKTEST_CFG = config['T_TEST_BACKTEST']
        optimize_params_cfg = config['optimize_params']

        # Hawkes state for non-baseline models
        H_t_prev = np.zeros(num_targets) if config_name.startswith(('contagion', 'hybrid')) else None

        current_T_TRAIN_OPT = config.get('T_TRAIN_OPT', 8)
        current_T_TEST_OPT = config.get('T_TEST_OPT', 1)

        start_index_cfg = T_TRAIN_BACKTEST_CFG
        end_index_cfg = T_TRAIN_BACKTEST_CFG + T_TEST_BACKTEST_CFG
        actual_start_index = max(start_index_cfg, current_T_TRAIN_OPT + current_T_TEST_OPT) if optimize_params_cfg else start_index_cfg
        if len(Y_t_full) < end_index_cfg or len(Y_t_full) < actual_start_index + 1:
            continue

        # Collect predictions and labels per target
        preds_per_target: List[List[float]] = [[] for _ in range(num_targets)]
        trues_per_target: List[List[int]] = [[] for _ in range(num_targets)]

        for t in range(actual_start_index, end_index_cfg):
            Y_train = Y_t_full.iloc[t - T_TRAIN_BACKTEST_CFG: t]
            Y_obs_counts = Y_t_full.iloc[t]
            y_true_binary = (Y_obs_counts > 0).astype(int)

            optimal_decay = DECAY_FIXED
            optimal_jump = JUMP_FIXED
            if optimize_params_cfg:
                best_nll_for_step = float('inf')
                opt_start_index_full = t - current_T_TEST_OPT - current_T_TRAIN_OPT
                if opt_start_index_full >= 0:
                    for decay in np.arange(0.1, 1.0, 0.05):
                        for jump in np.arange(0.001, 0.201, 0.01):
                            current_H_prev_mini = np.zeros(num_targets)
                            mini_log_losses = []
                            for opt_t_relative in range(current_T_TRAIN_OPT, current_T_TRAIN_OPT + current_T_TEST_OPT):
                                opt_t_full = opt_start_index_full + opt_t_relative
                                Y_train_mini = Y_t_full.iloc[opt_t_full - current_T_TRAIN_OPT: opt_t_full]
                                Y_obs_counts_mini = Y_t_full.iloc[opt_t_full].values
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
                                avg_nll_mini = float(np.mean(mini_log_losses))
                                if avg_nll_mini < best_nll_for_step:
                                    best_nll_for_step = avg_nll_mini
                                    optimal_decay = decay
                                    optimal_jump = jump

            # Forecast for the main step
            if config_name.startswith('baseline'):
                P_forecast = func(Y_train)
                p = P_forecast.values
            else:
                p, H_t_prev = func(
                    Y_train=Y_train,
                    Y_current=Y_obs_counts.values,
                    H_t_prev=H_t_prev,
                    decay=optimal_decay,
                    jump=optimal_jump
                )
            y = y_true_binary.values

            # Append per-target
            for i in range(num_targets):
                preds_per_target[i].append(float(p[i]))
                trues_per_target[i].append(int(y[i]))

        # Flatten for global calibration
        pred_all = np.array([pp for sub in preds_per_target for pp in sub], dtype=float)
        true_all = np.array([tt for sub in trues_per_target for tt in sub], dtype=int)

        stats = _compute_calibration_stats_adaptive(pred_all, true_all, n_bins=n_bins, min_count_per_bin=min_count_per_bin)
        # Prepare calibrated predictions if selection is available
        cal_stats = None
        cal_label = None
        p_cal = None
        method = None
        T = None
        s_val = None
        sel = selection.get(config_name)
        if sel and sel.get('method') in ('histogram','isotonic','temperature','intensity'):
            method = sel['method']
            # Build calibrated predictions
            if method in ('histogram','isotonic'):
                # build smoothed event rates from raw bins
                centers = np.array(stats['bin_centers'], dtype=float)
                counts = np.array(stats['bin_counts'], dtype=float)
                event_rates = np.array(stats['bin_event_rate'], dtype=float)
                event_rates_sm = _apply_laplace_smoothing(event_rates, counts, alpha=0.5)
                if method == 'isotonic':
                    try:
                        x_fit, y_fit = _fit_isotonic_mapping(centers, event_rates_sm, counts)
                        p_cal = _apply_isotonic(pred_all, x_fit, y_fit)
                    except Exception:
                        diff = np.abs(pred_all[:, None] - centers[None, :])
                        idx = np.argmin(diff, axis=1)
                        p_cal = event_rates_sm[idx]
                else:
                    diff = np.abs(pred_all[:, None] - centers[None, :])
                    idx = np.argmin(diff, axis=1)
                    p_cal = event_rates_sm[idx]
                cal_label = f"Calibrated: {method}"
            elif method == 'temperature':
                eps = 1e-12
                p = np.clip(pred_all, eps, 1 - eps)
                logits = np.log(p) - np.log(1 - p)
                T = float(sel.get('T') or 1.0)
                p_cal = 1.0 / (1.0 + np.exp(-(logits / max(T, eps))))
                cal_label = f"Calibrated: temperature (T={T:.2f})"
            elif method == 'intensity':
                eps = 1e-12
                lam = -np.log(np.clip(1 - pred_all, eps, 1.0))
                s_val = float(sel.get('s') or 1.0)
                p_cal = 1.0 - np.exp(-(s_val * lam))
                cal_label = f"Calibrated: intensity (s={s_val:.2f})"
            else:
                p_cal = None
            if p_cal is not None:
                cal_stats = _compute_calibration_stats_adaptive(p_cal, true_all, n_bins=n_bins, min_count_per_bin=min_count_per_bin)
        # Average Negative Log-Likelihood per prediction (for delta vs baseline reporting)
        try:
            total_nll = float(calculate_log_loss(true_all, np.clip(pred_all, 1e-15, 1 - 1e-15)))
            avg_nll = total_nll / max(1, len(pred_all))
        except Exception:
            avg_nll = None
        # Post-calibration NLL
        try:
            if p_cal is not None:
                total_nll_cal = float(calculate_log_loss(true_all, np.clip(p_cal, 1e-15, 1 - 1e-15)))
                avg_nll_cal = total_nll_cal / max(1, len(p_cal))
            else:
                avg_nll_cal = None
        except Exception:
            avg_nll_cal = None
        slug = config_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
        # Save plot (overlay if we have calibrated stats)
        _save_reliability_plot(stats, os.path.join(output_dir, f'reliability_{slug}.png'), f'Reliability: {config_name}', stats_cal=cal_stats, cal_label=(cal_label or 'Calibrated'))
        # Save probability histogram (raw + calibrated overlay if available)
        _save_probability_histogram(pred_all, os.path.join(output_dir, f'histogram_{slug}.png'), f'Probability Histogram: {config_name}', pred_cal=p_cal if 'p_cal' in locals() else None)
        # Save per-bin CSV
        bins_csv = os.path.join(output_dir, f'bins_{slug}.csv')
        with open(bins_csv, 'w', encoding='utf-8') as f:
            f.write('bin_center,bin_count,mean_pred,event_rate,event_rate_lo95,event_rate_hi95\n')
            for c, cnt, mp, er in zip(stats['bin_centers'], stats['bin_counts'], stats['bin_mean_pred'], stats['bin_event_rate']):
                lo, hi = _wilson_ci(float(er), int(cnt))
                f.write(f"{c},{cnt},{mp},{er},{lo},{hi}\n")
        # Append summary
        summary_rows.append({
            'model_config': config_name,
            'test_window_weeks': T_TEST_BACKTEST_CFG,
            # Backward compatible raw fields (legacy consumers use these)
            'ece': stats['ece'],
            'brier': stats['brier'],
            'nll': avg_nll,
            'n_predictions': int(len(pred_all)),
            # Explicit raw vs calibrated metrics
            'ece_raw': stats['ece'],
            'brier_raw': stats['brier'],
            'nll_raw': avg_nll,
            'ece_cal': (cal_stats['ece'] if cal_stats else None),
            'brier_cal': (cal_stats['brier'] if cal_stats else None),
            'nll_cal': avg_nll_cal,
            # Method used for post-cal metrics and parameters, if any
            'calibration_method': method,
            'temperature_T': (float(T) if T is not None else None),
            'intensity_s': (float(s_val) if s_val is not None else None)
        })

        # Per-segment breakdown (Bundesland / Sector)
        # Build index lists per segment
        bundesland_to_indices: Dict[str, List[int]] = {}
        sector_to_indices: Dict[str, List[int]] = {}
        for i, label in enumerate(target_labels):
            parts = [p.strip() for p in label.split('|')]
            if len(parts) == 2:
                bl, sec = parts[0], parts[1]
            else:
                # Fallback parsing in case of different delimiter use
                tokens = label.split(' | ')
                bl = tokens[0]
                sec = tokens[1] if len(tokens) > 1 else 'Other'
            bundesland_to_indices.setdefault(bl, []).append(i)
            sector_to_indices.setdefault(sec, []).append(i)

        # Convert per-target lists to arrays of shape (targets, samples_per_target)
        # then index-select and flatten
        preds_arr = [np.array(lst, dtype=float) for lst in preds_per_target]
        trues_arr = [np.array(lst, dtype=int) for lst in trues_per_target]

        def _gather(indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
            p_list, t_list = [], []
            for idx in indices:
                p_list.append(preds_arr[idx])
                t_list.append(trues_arr[idx])
            if not p_list:
                return np.array([]), np.array([])
            return np.concatenate(p_list), np.concatenate(t_list)

        # Bundesland
        for bl, idxs in bundesland_to_indices.items():
            p_seg, t_seg = _gather(idxs)
            if p_seg.size == 0:
                continue
            s = _compute_calibration_stats_adaptive(p_seg, t_seg, n_bins=n_bins, min_count_per_bin=min_count_per_bin)
            segment_rows.append({
                'model_config': config_name,
                'test_window_weeks': T_TEST_BACKTEST_CFG,
                'segment_type': 'Bundesland',
                'segment': bl,
                'ece': s['ece'],
                'brier': s['brier'],
                'n_predictions': int(p_seg.size)
            })

        # Sector
        for sec, idxs in sector_to_indices.items():
            p_seg, t_seg = _gather(idxs)
            if p_seg.size == 0:
                continue
            s = _compute_calibration_stats_adaptive(p_seg, t_seg, n_bins=n_bins, min_count_per_bin=min_count_per_bin)
            segment_rows.append({
                'model_config': config_name,
                'test_window_weeks': T_TEST_BACKTEST_CFG,
                'segment_type': 'Sector',
                'segment': sec,
                'ece': s['ece'],
                'brier': s['brier'],
                'n_predictions': int(p_seg.size)
            })

    # Write summaries
    with open(os.path.join(output_dir, 'calibration_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, indent=2)

    # Write per-segment CSV
    seg_csv = os.path.join(output_dir, 'segment_breakdown.csv')
    with open(seg_csv, 'w', encoding='utf-8') as f:
        f.write('model_config,test_window_weeks,segment_type,segment,ece,brier,n_predictions\n')
        for row in segment_rows:
            f.write(
                f"{row['model_config']},{row['test_window_weeks']},{row['segment_type']},"
                f"{row['segment']},{row['ece']},{row['brier']},{row['n_predictions']}\n"
            )

    print(f"Calibration artifacts written to: {os.path.abspath(output_dir)}")


def run_calibration_method_comparison(
    model_name: str = 'all',
    data_file: str = 'data/incidents.csv',
    n_bins: int = 10,
    laplace_alpha: float = 0.5,
    min_count_per_bin: int = 100,
    output_dir: str = 'output/calibration'
):
    """Compare calibration methods for each model/horizon.

    Methods compared:
      - histogram (adaptive bins + Laplace smoothing)
      - isotonic (fit on binned points with counts)
      - temperature scaling (logit temperature) with T grid search
      - intensity scaling (Poisson intensity scale) with s grid search

    Writes output to calibration_method_comparison.json with per-model metrics and recommended default (by ECE then Brier),
    plus overall averages across methods.
    """
    # Reuse configs from run_calibration_analysis
    model_configs = {}
    baseline_configs = {
        'baseline (8/1)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 8, 'T_TEST_BACKTEST': 1, 'optimize_params': False},
        'baseline (26/13)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 26, 'T_TEST_BACKTEST': 13, 'optimize_params': False},
        'baseline (52/26)': {'func': baseline_forecast, 'H_prev': None, 'T_TRAIN_BACKTEST': 52, 'T_TEST_BACKTEST': 26, 'optimize_params': False},
    }
    if model_name.lower() in ['all', 'baseline']:
        model_configs.update(baseline_configs)
    if model_name.lower() in ['all', 'contagion']:
        model_configs['contagion (8/1 Opt)'] = {
            'func': contagion_only_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 8,
            'T_TEST_BACKTEST': 1,
            'optimize_params': True,
            'T_TRAIN_OPT': 4,
            'T_TEST_OPT': 1
        }
    if model_name.lower() in ['all', 'hybrid']:
        model_configs['hybrid (26/13 Opt)'] = {
            'func': hybrid_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 26,
            'T_TEST_BACKTEST': 13,
            'optimize_params': True,
            'T_TRAIN_OPT': 16,
            'T_TEST_OPT': 4
        }
        model_configs['hybrid (52/26 Opt)'] = {
            'func': hybrid_forecast,
            'H_prev': None,
            'T_TRAIN_BACKTEST': 52,
            'T_TEST_BACKTEST': 26,
            'optimize_params': True,
            'T_TRAIN_OPT': 4,
            'T_TEST_OPT': 1
        }

    max_train_backtest = max([c['T_TRAIN_BACKTEST'] for c in model_configs.values()])
    max_test_backtest = max([c['T_TEST_BACKTEST'] for c in model_configs.values()])
    max_train_opt = max([c.get('T_TRAIN_OPT', 0) for c in model_configs.values() if c.get('optimize_params', False)], default=0)
    max_test_opt = max([c.get('T_TEST_OPT', 0) for c in model_configs.values() if c.get('optimize_params', False)], default=0)
    earliest_start_offset = max(max_train_backtest, max_train_opt + max_test_opt)
    latest_end_index = max([c['T_TRAIN_BACKTEST'] + c['T_TEST_BACKTEST'] for c in model_configs.values()])

    Y_t_full, num_targets, total_weeks, _ = prepare_data(data_file, earliest_start_offset, latest_end_index)

    os.makedirs(output_dir, exist_ok=True)
    comparison_rows = []

    for config_name, config in model_configs.items():
        func = config['func']
        T_TRAIN_BACKTEST_CFG = config['T_TRAIN_BACKTEST']
        T_TEST_BACKTEST_CFG = config['T_TEST_BACKTEST']
        optimize_params_cfg = config['optimize_params']
        H_t_prev = np.zeros(num_targets) if config_name.startswith(('contagion', 'hybrid')) else None

        current_T_TRAIN_OPT = config.get('T_TRAIN_OPT', 8)
        current_T_TEST_OPT = config.get('T_TEST_OPT', 1)
        start_index_cfg = T_TRAIN_BACKTEST_CFG
        end_index_cfg = T_TRAIN_BACKTEST_CFG + T_TEST_BACKTEST_CFG
        actual_start_index = max(start_index_cfg, current_T_TRAIN_OPT + current_T_TEST_OPT) if optimize_params_cfg else start_index_cfg
        if len(Y_t_full) < end_index_cfg or len(Y_t_full) < actual_start_index + 1:
            continue

        preds_per_target: List[List[float]] = [[] for _ in range(num_targets)]
        trues_per_target: List[List[int]] = [[] for _ in range(num_targets)]

        for t in range(actual_start_index, end_index_cfg):
            Y_train = Y_t_full.iloc[t - T_TRAIN_BACKTEST_CFG: t]
            Y_obs_counts = Y_t_full.iloc[t]
            y_true_binary = (Y_obs_counts > 0).astype(int)

            optimal_decay = DECAY_FIXED
            optimal_jump = JUMP_FIXED
            if optimize_params_cfg:
                best_nll_for_step = float('inf')
                opt_start_index_full = t - current_T_TEST_OPT - current_T_TRAIN_OPT
                if opt_start_index_full >= 0:
                    for decay in np.arange(0.1, 1.0, 0.05):
                        for jump in np.arange(0.001, 0.201, 0.01):
                            current_H_prev_mini = np.zeros(num_targets)
                            mini_log_losses = []
                            for opt_t_relative in range(current_T_TRAIN_OPT, current_T_TRAIN_OPT + current_T_TEST_OPT):
                                opt_t_full = opt_start_index_full + opt_t_relative
                                Y_train_mini = Y_t_full.iloc[opt_t_full - current_T_TRAIN_OPT: opt_t_full]
                                Y_obs_counts_mini = Y_t_full.iloc[opt_t_full].values
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
                                avg_nll_mini = float(np.mean(mini_log_losses))
                                if avg_nll_mini < best_nll_for_step:
                                    best_nll_for_step = avg_nll_mini
                                    optimal_decay = decay
                                    optimal_jump = jump

            if config_name.startswith('baseline'):
                P_forecast = func(Y_train)
                p = P_forecast.values
            else:
                p, H_t_prev = func(
                    Y_train=Y_train,
                    Y_current=Y_obs_counts.values,
                    H_t_prev=H_t_prev,
                    decay=optimal_decay,
                    jump=optimal_jump
                )
            y = y_true_binary.values

            for i in range(num_targets):
                preds_per_target[i].append(float(p[i]))
                trues_per_target[i].append(int(y[i]))

        pred_all = np.array([pp for sub in preds_per_target for pp in sub], dtype=float)
        true_all = np.array([tt for sub in trues_per_target for tt in sub], dtype=int)

        # Compute baseline stats and bins (edges, centers, counts, event rates)
        stats = _compute_calibration_stats_adaptive(pred_all, true_all, n_bins=n_bins, min_count_per_bin=min_count_per_bin)
        centers = np.array(stats['bin_centers'], dtype=float)
        counts = np.array(stats['bin_counts'], dtype=float)
        event_rates = np.array(stats['bin_event_rate'], dtype=float)
        # Smoothing
        event_rates_sm = _apply_laplace_smoothing(event_rates, counts, alpha=laplace_alpha)
        # Histogram mapping by nearest center (aligns with forecast implementation)
        diff = np.abs(pred_all[:, None] - centers[None, :])
        idx = np.argmin(diff, axis=1)
        p_hist = event_rates_sm[idx]
        # Isotonic: fit on centers->smoothed rates weighted by counts
        try:
            x_fit, y_fit = _fit_isotonic_mapping(centers, event_rates_sm, counts)
            p_iso = _apply_isotonic(pred_all, x_fit, y_fit)
        except Exception:
            p_iso = p_hist.copy()

        # Temperature scaling: p' = sigmoid(logit(p)/T), grid search T
        eps = 1e-12
        def _logit(x):
            x = np.clip(x, eps, 1 - eps)
            return np.log(x) - np.log(1 - x)
        def _sigmoid(z):
            return 1 / (1 + np.exp(-z))
        logits = _logit(np.clip(pred_all, eps, 1 - eps))
        T_grid = np.arange(0.5, 2.01, 0.05)
        best_T = None
        best_nll = float('inf')
        for T in T_grid:
            p_t = _sigmoid(logits / T)
            nll = -float(np.mean(true_all * np.log(np.clip(p_t, eps, 1 - eps)) + (1 - true_all) * np.log(np.clip(1 - p_t, eps, 1 - eps))))
            if nll < best_nll:
                best_nll = nll
                best_T = float(T)
        p_temp = _sigmoid(logits / best_T) if best_T is not None else np.clip(pred_all, eps, 1 - eps)

        # Intensity scaling: invert Poisson prob to lambda, scale, re-convert
        lam = -np.log(np.clip(1 - pred_all, eps, 1.0))
        s_grid = np.arange(0.5, 3.01, 0.05)
        best_s = None
        best_nll_s = float('inf')
        for s in s_grid:
            lam_s = s * lam
            p_s = 1 - np.exp(-lam_s)
            nll = -float(np.mean(true_all * np.log(np.clip(p_s, eps, 1 - eps)) + (1 - true_all) * np.log(np.clip(1 - p_s, eps, 1 - eps))))
            if nll < best_nll_s:
                best_nll_s = nll
                best_s = float(s)
        lam_s = (best_s if best_s is not None else 1.0) * lam
        p_int = 1 - np.exp(-lam_s)

        # Metrics
        def _metrics(p):
            brier = float(np.mean((p - true_all) ** 2))
            s = _compute_calibration_stats_adaptive(p, true_all, n_bins=n_bins, min_count_per_bin=min_count_per_bin)
            return s['ece'], brier
        ece_hist, brier_hist = _metrics(p_hist)
        ece_iso, brier_iso = _metrics(p_iso)
        ece_temp, brier_temp = _metrics(p_temp)
        ece_int, brier_int = _metrics(p_int)

        # Recommend per model by ECE then Brier
        candidates = [
            ('histogram', ece_hist, brier_hist),
            ('isotonic', ece_iso, brier_iso),
            ('temperature', ece_temp, brier_temp),
            ('intensity', ece_int, brier_int),
        ]
        # pick argmin by (ece, brier)
        recommended = min(candidates, key=lambda t: (t[1], t[2]))[0]

        comparison_rows.append({
            'model_config': config_name,
            'test_window_weeks': T_TEST_BACKTEST_CFG,
            'n_predictions': int(pred_all.size),
            'laplace_alpha': laplace_alpha,
            'histogram': {'ece': ece_hist, 'brier': brier_hist},
            'isotonic': {'ece': ece_iso, 'brier': brier_iso},
            'temperature': {'ece': ece_temp, 'brier': brier_temp, 'T': best_T},
            'intensity': {'ece': ece_int, 'brier': brier_int, 's': best_s},
            'recommended_method': recommended
        })

    # Overall recommendation by average ECE across models/horizons (tie-breaker: Brier)
    if comparison_rows:
        ece_avgs = {
            'histogram': float(np.mean([r['histogram']['ece'] for r in comparison_rows])),
            'isotonic': float(np.mean([r['isotonic']['ece'] for r in comparison_rows])),
            'temperature': float(np.mean([r['temperature']['ece'] for r in comparison_rows])),
            'intensity': float(np.mean([r['intensity']['ece'] for r in comparison_rows]))
        }
        brier_avgs = {
            'histogram': float(np.mean([r['histogram']['brier'] for r in comparison_rows])),
            'isotonic': float(np.mean([r['isotonic']['brier'] for r in comparison_rows])),
            'temperature': float(np.mean([r['temperature']['brier'] for r in comparison_rows])),
            'intensity': float(np.mean([r['intensity']['brier'] for r in comparison_rows]))
        }
        overall = min(ece_avgs.keys(), key=lambda m: (ece_avgs[m], brier_avgs[m]))
    else:
        overall = None

    out = {
        'n_bins': n_bins,
        'laplace_alpha': laplace_alpha,
        'min_count_per_bin': min_count_per_bin,
        'overall_recommended_method': overall,
        'overall_metrics': {
            'histogram': {'ece_avg': ece_avgs['histogram'] if comparison_rows else None, 'brier_avg': brier_avgs['histogram'] if comparison_rows else None},
            'isotonic': {'ece_avg': ece_avgs['isotonic'] if comparison_rows else None, 'brier_avg': brier_avgs['isotonic'] if comparison_rows else None},
            'temperature': {'ece_avg': ece_avgs['temperature'] if comparison_rows else None, 'brier_avg': brier_avgs['temperature'] if comparison_rows else None},
            'intensity': {'ece_avg': ece_avgs['intensity'] if comparison_rows else None, 'brier_avg': brier_avgs['intensity'] if comparison_rows else None}
        },
        'per_model': comparison_rows
    }

    with open(os.path.join(output_dir, 'calibration_method_comparison.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f"Calibration method comparison written to: {os.path.abspath(os.path.join(output_dir, 'calibration_method_comparison.json'))}")

# --- 5. EXECUTION EXAMPLE ---
# Note: In Colab, you need to upload the CSV file first, e.g., 'incidents-updated-geo-sector.csv'

# Example Usage with new parameters:
# do_backtest(model_name, data_file, decay_values, jump_values, T_TRAIN_OPT, T_TEST_OPT)
#
# Parameters:
#   model_name (str): Choose 'baseline', 'contagion', 'hybrid', or 'all'.
#   data_file (str): The path to the CSV file containing the incident data. Defaults to 'data/incidents.csv'.
#   decay_values (np.ndarray): Array of potential values for DECAY_FIXED in the optimization sweep. Defaults to np.arange(0.1, 1.0, 0.05).
#   jump_values (np.ndarray): Array of potential values for JUMP_FIXED in the optimization sweep. Defaults to np.arange(0.001, 0.201, 0.01).
#   T_TRAIN_OPT (int): The number of weeks for the optimization training window. Defaults to 8. (These defaults are overridden by config)
#   T_TEST_OPT (int): The number of weeks for the optimization test period. Defaults to 1. (These defaults are overridden by config)
#
# Returns:
#   A DataFrame with Log Loss results for the selected model(s).

# Example: Run ALL models with their specified configurations and optimization
if __name__ == "__main__":
    results_all_configured = do_backtest('all', 'data/incidents.csv')
                                         #decay_values=np.arange(0.1, 1.0, 0.05), # Example optimization ranges
                                         #jump_values=np.arange(0.001, 0.201, 0.01)) # T_TRAIN_OPT and T_TEST_OPT are now defaults in config