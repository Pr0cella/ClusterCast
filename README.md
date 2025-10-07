# ClusterCast: Ransomware Risk Forecasting System

## Table of Contents
- [Executive Summary](#executive-summary)
- [Mathematical Foundation and Model Selection](#mathematical-foundation-and-model-selection)
- [Probabilistic Distribution Analysis and Accuracy Assessment](#probabilistic-distribution-analysis-and-accuracy-assessment)
- [Parameter Optimization and Model Configuration](#parameter-optimization-and-model-configuration)
- [Sparse Data Influence on Model Performance](#sparse-data-influence-on-model-performance)
- [System Architecture and Implementation](#system-architecture-and-implementation)
  - [Modular framework and plugins](#modular-framework-and-plugins)
  - [Scaffold a plugin model (CLI)](#scaffold-a-plugin-model-cli)
  - [Rolling backtests for modular models](#rolling-backtests-for-modular-models)
  - [Compare two backtest runs (delta summary)](#compare-two-backtest-runs-delta-summary)
  - [Execution Parameters](#execution-parameters)
  - [Data Requirements](#data-requirements)
  - [Output Specifications](#output-specifications)
- [Dependencies and Technical Requirements](#dependencies-and-technical-requirements)
- [Calibration](#calibration)
- [Validation and Quality Assurance](#validation-and-quality-assurance)
- [JSON Report Fields (Calibration-aware)](#json-report-fields-calibration-aware)
- [Calibration options and robustness](#calibration-options-and-robustness)

## Executive Summary

This system implements advanced probabilistic forecasting models to predict ransomware attack risks across German federal states (Bundesländer) and economic sectors. The framework combines historical incident data with sophisticated mathematical models to generate actionable risk assessments for cybersecurity decision-making.

**Core Approach**: The system employs three complementary statistical models: (1) Historical Frequency Baseline, (2) Contagion-Only Model based on Hawkes processes, and (3) Hybrid Model combining target-specific baselines with contagion dynamics. These models capture both the persistent strategic targeting patterns of threat actors and the temporal clustering effects characteristic of ransomware campaigns.

**Data Integration**: Historical ransomware incidents are aggregated by geographic region (Bundesland) and economic sector, creating target nodes representing unique Bundesland-Sector combinations. Weekly incident counts serve as the fundamental time series data for model training and forecasting.

**Predictive Output**: The system generates probabilistic forecasts indicating the likelihood of at least one ransomware incident occurring within specified time horizons for each target combination. Risk bands (Very High, High, Medium, Low, Very Low) translate numerical probabilities into actionable intelligence for security teams.

**Parameter Optimization**: Automated parameter optimization employs rolling backtesting to identify optimal model configurations, ensuring predictions adapt to evolving threat landscapes while maintaining statistical rigor.

**Modular framework**: Built for rapid iteration and reproducibility, the framework brings together forecasting, calibration‑aware reporting, rolling backtests, and parameter search under a plugin‑friendly interface. Teams can integrate new statistical models quickly, benchmark against baselines and prior runs, and compare outputs across datasets. **While our contagion and hybrid models show promising backtests, they may not generalize to every real‑world setting.** To encourage rigor and adaptability, the framework invites contributions from the open‑source and scientific community and is straightforward to adapt beyond the current German Bundesland × sector use case-the same interfaces apply to other labeled panel time series with minimal changes. See [Modular framework and plugins](#modular-framework-and-plugins).

## Mathematical Foundation and Model Selection

### 1. Baseline Model: Historical Frequency Approach

**Mathematical Formulation**:
```
λᵥ = (1/T) × Σ(Yᵥ(t))  for t ∈ [T_train]
P(X ≥ 1) = 1 - exp(-λᵥ)
```

Where:
- `v`: target index (Bundesland × Sector combination)
- `Yᵥ(t)`: weekly incident count for target `v` at week `t`
- `T`: number of weeks in the training window
- `λᵥ`: Poisson intensity (expected incidents per week) for target `v`
- Probability mapping: assuming weekly counts `Xᵥ ~ Poisson(λᵥ)`, the probability of at least one incident in a week is `P(Xᵥ ≥ 1) = 1 - exp(-λᵥ)`

**Rationale**: Ransomware threat actors exhibit persistent targeting preferences based on sector vulnerability, technological infrastructure, and economic value. The baseline model captures these strategic patterns by computing the historical average incident rate for each Bundesland-Sector combination.

**Advantages**:
- **Interpretability**: Direct relationship between historical frequency and predicted risk
- **Stability**: Robust to short-term fluctuations and data anomalies
- **Computational Efficiency**: Minimal computational overhead for real-time applications
- **Baseline Reference**: Provides performance benchmark for more sophisticated models

**Limitations**:
- **Static Assumptions**: Cannot adapt to evolving threat patterns or campaign cycles
- **No Temporal Dependencies**: Ignores clustering effects and contagion dynamics
- **Limited Predictive Power**: May underperform during coordinated attack campaigns

### 2. Contagion-Only Model: Hawkes Process Implementation

**Mathematical Formulation**:
```
λᵥ(t) = μ_global + Hᵥ(t)
Hᵥ(t) = α × Hᵥ(t-1) + β × Yᵥ(t-1)
P(X ≥ 1) = 1 - exp(-λᵥ(t))
```

Where:
- `μ_global`: Global baseline intensity (average incident rate across all targets)
- `α` (DECAY): Memory decay parameter (0 < α < 1)
- `β` (JUMP): Excitation parameter (β > 0)
- `Hᵥ(t)`: Hawkes process capturing temporal clustering
- `v`: target index (Bundesland × Sector combination)
- `t`: discrete week index
- `Yᵥ(t)`: weekly incident count for target `v` at week `t`

**Rationale**: Ransomware campaigns often exhibit temporal clustering due to coordinated group activities, vulnerability exploitation windows, and copycat effects. Hawkes processes mathematically model these self-exciting properties where past events increase the probability of future events.

**Advantages**:
- **Temporal Dynamics**: Captures short-term clustering and campaign effects
- **Adaptive Response**: Responds dynamically to recent incident patterns
- **Mathematical Rigor**: Well-established stochastic process theory foundation
- **Campaign Detection**: Sensitive to coordinated attack patterns

**Implementation Details**:
- **DECAY Parameter (α)**: Controls memory persistence (typical range: 0.1-1.0)
  - Higher values: Longer memory of past incidents
  - Lower values: Rapid forgetting of historical patterns
- **JUMP Parameter (β)**: Controls excitation magnitude (typical range: 0.001-0.201)
  - Higher values: Stronger contagion effects
  - Lower values: Minimal clustering response

### 3. Hybrid Model: Strategic Baseline + Hawkes Process

**Mathematical Formulation**:
```
λᵥ(t) = μᵥ + Hᵥ(t)
μᵥ = (1/T) × Σ(Yᵥ(t))  for t ∈ [T_train]
Hᵥ(t) = α × Hᵥ(t-1) + β × Yᵥ(t-1)
P(X ≥ 1) = 1 - exp(-λᵥ(t))
```

Where:
- `v`: target index (Bundesland × Sector combination)
- `t`: discrete week index
- `μᵥ`: target-specific baseline intensity (mean incidents per week over the training window)
- `Hᵥ(t)`: self-exciting memory state at time `t`
- `α` (DECAY): memory persistence (0 < α < 1)
- `β` (JUMP): excitation magnitude (β > 0)
- `Yᵥ(t-1)`: incident count in the previous week
- `λᵥ(t)`: total intensity at time `t`
- Probability mapping: with weekly counts `Xᵥ(t) ~ Poisson(λᵥ(t))`, `P(Xᵥ(t) ≥ 1) = 1 - exp(-λᵥ(t))`

**Rationale**: This model synthesizes the strategic targeting preferences captured by target-specific baselines with the temporal clustering dynamics of Hawkes processes. It recognizes that different Bundesland-Sector combinations have inherent risk levels while remaining susceptible to campaign-driven clustering effects.

**Advantages**:
- **Comprehensive Modeling**: Captures both strategic and temporal risk factors
- **Target Specificity**: Maintains individual baseline risk assessments
- **Dynamic Adaptation**: Responds to both long-term trends and short-term clusters
- **Balanced Approach**: Combines interpretability of baselines with sophistication of temporal modeling

## Probabilistic Distribution Analysis and Accuracy Assessment

### Statistical Foundation

The system employs **Poisson regression** as the underlying probabilistic framework, where incident counts follow a Poisson distribution with time-varying intensity parameters. This choice is mathematically justified for several reasons:

1. **Discrete Event Nature**: Ransomware incidents are discrete occurrences
2. **Rare Event Properties**: Low base rates characteristic of Poisson processes
3. **Independence Assumptions**: Individual incidents are conditionally independent given the intensity
4. **Temporal Flexibility**: Intensity parameters can incorporate complex temporal dependencies

### Accuracy Metrics and Validation

**Primary Metric: Negative Log-Likelihood (NLL)**
```
NLL = -Σ[y_true × log(p_pred) + (1 - y_true) × log(1 - p_pred)]
```

This metric is selected because:
- **Proper Scoring Rule**: Provides unbiased assessment of probabilistic predictions
- **Calibration Sensitivity**: Penalizes both over-confidence and under-confidence
- **Binary Classification**: Adapted for "incident/no incident" prediction tasks

[Acknowledgement](https://psymbio.github.io/posts/NLLLoss/)

**Brier Score**
```
Brier = mean (p_pred - y_true)^2
```
- Measures the mean squared error of probabilistic predictions.
- Proper scoring rule for binary outcomes; lower is better.
- Sensitive to both calibration (alignment with frequencies) and sharpness (concentration away from 0.5).

**Expected Calibration Error (ECE)**
```
ECE = Σ_i (n_i / N) · | mean(p_pred)_i - event_rate_i |
```
- Aggregates calibration gaps across probability bins; lower indicates better calibration.
- `i` indexes bins, `n_i` is bin count, `N` is total samples.
- We typically use 10 bins with binning by predicted probability; variants (ECE_k, adaptive bins) are also used in practice.
- Cross-reference: See [Calibration](#calibration) for operational details (reliability diagrams, method selection, and artifacts).

Bin choices in practice:
- Backtests: fixed 10 equal-width probability bins for headline ECE.
- Calibration analysis: adaptive quantile bins merged to satisfy `CAL_MIN_COUNT_PER_BIN` (see Calibration section), improving stability in sparse regimes.

**Performance Validation Framework**:
- **Rolling Backtesting**: Sequential model evaluation mimicking real-world deployment
- **Out-of-Sample Testing**: Strict temporal separation between training and testing data
- **Cross-Validation**: Multiple forecast horizons (1, 13, 26 weeks) to assess temporal stability
- **Skill Score Calculation**: Relative performance improvement over baseline models

### Interpretability Framework

**Risk Band Translation**:
The system translates numerical probabilities into interpretable risk categories:

| Risk Band | Threshold | Interpretation |
|-----------|-----------|----------------|
| Very High | ≥ 10.0% | Immediate review and heightened monitoring |
| High | 5.0% - 9.9% | Proactive assessment and targeted defenses |
| Medium | 2.0% - 4.9% | Regular monitoring and standard practices |
| Low | 0.5% - 1.9% | Baseline security measures sufficient |
| Very Low | < 0.5% | Minimal predicted risk |

**Confidence Intervals**: While point estimates provide primary guidance, the Poisson framework enables confidence interval calculation for uncertainty quantification.

**Feature Attribution**: Target-specific baseline intensities and Hawkes memory states provide transparent insight into risk factor contributions.

## Parameter Optimization and Model Configuration

### Optimization Framework

**Search Space Definition**:
- **DECAY Parameter**: [0.1, 1.0] with 0.05 step size (18 values)
- **JUMP Parameter**: [0.001, 0.201] with 0.01 step size (20 values)
- **Total Combinations**: 360 parameter configurations

**Optimization Strategy**:
```
For each forecast step t:
  1. Define optimization window: [t - T_TEST_OPT - T_TRAIN_OPT, t - T_TEST_OPT]
  2. For each (DECAY, JUMP) combination:
     a. Train model on optimization training window
     b. Forecast on optimization test window
     c. Calculate average NLL
  3. Select parameters minimizing NLL
  4. Apply optimal parameters to main forecast step
```

**Window Configuration**:
- **Contagion Model (8/1)**: Optimization window 4 weeks training, 1 week testing
- **Hybrid Model (26/13)**: Optimization window 16 weeks training, 4 weeks testing
- **Hybrid Model (52/26)**: Optimization window 4 weeks training, 1 week testing

### Current Optimal Parameters

Based on recent optimization runs:
- **DECAY**: 0.95 (indicating strong memory persistence)
- **JUMP**: 0.19 (suggesting significant contagion effects)

These values indicate that the German ransomware landscape exhibits:
1. **Long Memory**: Past incidents influence risk for extended periods
2. **Strong Clustering**: Individual incidents substantially increase short-term probabilities

## Sparse Data Influence on Model Performance

### Challenge Definition

**Sparse Data Characteristics**:
- **Temporal Sparsity**: Most Bundesland-Sector combinations experience zero incidents in most time periods
- **Spatial Sparsity**: Incident concentration in specific geographic and sectoral clusters
- **Class Imbalance**: Approximately 95-98% of observations are zero-incident weeks

### Impact on Model Components

**1. Baseline Model Response**:
- **Advantage**: Robust to sparsity through averaging across training windows
- **Challenge**: May underestimate risk for targets with few historical incidents
- **Mitigation**: Sufficient training window length (8-52 weeks) provides statistical stability

**2. Hawkes Process Sensitivity**:
- **Memory Initialization**: Sparse targets start with zero Hawkes memory
- **Excitation Dynamics**: Single incidents can dramatically increase subsequent probabilities
- **Decay Behavior**: Memory fades based on DECAY parameter regardless of subsequent activity

**3. Optimization Reliability**:
- **Parameter Stability**: Large search space (360 combinations) provides robustness
- **Local Minima**: Multiple optimization windows reduce overfitting risk
- **Validation Challenges**: Short optimization windows may be noisy for sparse targets

### Sparse Data Mitigation Strategies

**1. Hierarchical Smoothing**:
- Global baseline component in Contagion model provides stability
- Target-specific baselines in Hybrid model maintain individual risk profiles

**2. Regularization Through Optimization**:
- Rolling parameter optimization prevents overfitting to sparse patterns
- Multiple forecast horizons validate temporal generalization

**3. Risk Band Aggregation**:
- Combined risk calculations at Bundesland and Sector levels provide portfolio-level insights
- Threshold filtering (0.5% minimum probability) focuses attention on actionable risks

**4. Uncertainty Communication**:
- Risk band classifications acknowledge inherent uncertainty in sparse regime predictions
- Skill score reporting provides context relative to baseline performance

## System Architecture and Implementation

### File Structure
```
src/
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
```

Notes:
- Split outputs are now the default. The top-level output folder contains exactly four primary files (predictions/calibration in md/json). All auxiliary calibration artifacts reside in output/calibration_assets/.
- A temporary read fallback to the legacy output/calibration/ path exists for one bake-in release. This may be removed later.

### Modular framework and plugins

This repository provides a pluggable model framework with a small runner and reporter integration:

- Interfaces and types live in `framework/base.py`.
- Built-in model adapters (`baseline`, `contagion`, `hybrid`) live in `framework/adapters.py`-they wrap the existing `forecast.py` functions.
- Plugin discovery loads any `ModelInterface` subclasses from `src/plugins/models/` at runtime.
- The modular runner (`framework/modular_runner.py`) lets you run any model (built-in or plugin), optionally optimize it, and emit the same split reports used in the main pipeline.

Run the modular runner:

```powershell
# List all available models (built-ins + plugins)
python -m framework.modular_runner --list-models

# Run a built-in model (hybrid) with small optimization
python -m framework.modular_runner --model hybrid --horizon 13 --train-window 26 --opt-budget 3

# Emit split reports (predictions/calibration)
python -m framework.modular_runner --model hybrid --horizon 13 --train-window 26 --opt-budget 2 --emit-reports

# Run a plugin model (example id: custom_example)
python -m framework.modular_runner --model custom_example --horizon 13 --train-window 26 --opt-budget 5 --emit-reports
```

Flags:
- `--plugins-dir`: Directory to search for plugin models (defaults to `src/plugins/models`).
- `--list-models`: Print all model ids and exit.
- `--model`: Model id to run (one of built-ins or discovered plugin ids).
- `--emit-reports`: Write split reports using `SplitReporter` and `forecast.py` generators.
- `--horizon`, `--train-window`, `--opt-budget`: Control forecasting and random search budget.

Create your own plugin model:

1) Add a file under `src/plugins/models/`, exporting a class that subclasses `ModelInterface`:

```python
# src/plugins/models/my_model.py
from typing import List, Any
import numpy as np
from framework.base import ModelInterface, SearchParam

class MyModel(ModelInterface):
  id = 'my_model'
  name = 'My Custom Model'
  version = '0.1.0'

  def required_features(self) -> List[str]:
    return ['week', 'Target_V']

  def get_search_space(self):
    return [SearchParam('param', kind='float', low=0.0, high=1.0, step=0.1)]

  def fit(self, train_df):
    # return any state needed for predict_proba
    baseline = 1 - np.exp(-train_df.mean(axis=0).values)
    return {'baseline': baseline}

  def predict_proba(self, state: Any, context: dict) -> np.ndarray:
    p = state['baseline']
    return np.clip(p, 0.0, 1.0)
```

2) Run `python -m framework.modular_runner --list-models` to see `my_model` registered, then run it with `--model my_model`.

No explicit registration is required-the loader auto-discovers subclasses.

### Scaffold a plugin model (CLI)

You can generate a ready-to-edit plugin file with the scaffold CLI. From the `src` directory in PowerShell:

```powershell
# Create src/plugins/models/my_model.py with class MyModel
python -m framework.scaffold_plugin --id my_model --name "My Model" --class-name MyModel

# Optional flags
#   --out-dir   Target directory (defaults to src/plugins/models)
#   --force     Overwrite existing file if present

# Discover your new model
python -m framework.modular_runner --list-models

# Run it
python -m framework.modular_runner --model my_model --horizon 13 --train-window 26 --opt-budget 3 --emit-reports
```

The generated file includes:
- `id`, `name`, `version` metadata
- `required_features()` and a simple `get_search_space()`
- Minimal `fit()` returning a baseline state and `predict_proba()` using it

Edit these methods to implement your logic; no registration changes are required.

### Rolling backtests for modular models

Use the modular backtest CLI to evaluate any built-in or plugin model with one-step-ahead rolling backtests and compare against the baseline.

Metrics reported:
- ECE (10-bin Expected Calibration Error)
- Brier score
- NLL (negative log-likelihood)
- Skill vs Baseline (% improvement vs historical-frequency baseline using same windows)

Examples (from the `src` directory in PowerShell):

```powershell
# Discover available models (built-ins + plugins)
python -m framework.modular_runner --list-models

# Backtest your plugin model over the last 13 weeks with a 26-week train window
python -m framework.modular_backtest --model my_model --train-window 26 --test-window 13 --opt-budget 10

# Backtest a built-in model (hybrid)
python -m framework.modular_backtest --model hybrid --train-window 26 --test-window 13 --opt-budget 20

# Use a specific data file (optional)
python -m framework.modular_backtest --model my_model --data "data\incidents.csv" --train-window 26 --test-window 13

# Backtest multiple models and export results (CSV/Markdown)
python -m framework.modular_backtest --model my_model --model hybrid --train-window 26 --test-window 13 --export-csv --export-md --out-dir output

# Backtest all discovered models
python -m framework.modular_backtest --all-models --train-window 26 --test-window 13 --export-csv --export-md --out-dir output
```

Notes:
- The backtest uses the same data preparation as the main pipeline (`forecast.prepare_data`).
- `--opt-budget` enables a small random search on a tiny pre-test validation slice to set model params.
- Works for any plugin that implements `ModelInterface` and is discoverable under `src/plugins/models/`.
- `--model` can be provided multiple times, or use `--all-models`.
- Use `--export-csv` and/or `--export-md` with `--out-dir` to save results; filenames include a timestamp for easy archiving.

### Compare two backtest runs (delta summary)

Use the compare utility to diff two exported backtest CSVs and produce per-model deltas and an aggregate summary.

Examples (from `src`, PowerShell):

```powershell
# Compare two runs and export delta tables (CSV/Markdown)
python -m framework.compare_backtests --old output\backtest_results_tw26_hw13_20251001-101500.csv --new output\backtest_results_tw26_hw13_20251006-090200.csv --export-csv --export-md --out-dir output

# Using glob patterns (picks the latest match for each)
python -m framework.compare_backtests --old "output\backtest_results_tw26_hw13_*.csv" --new "output\backtest_results_tw26_hw13_*.csv" --export-md --out-dir output

# Using directories (auto-picks previous vs latest in that folder)
python -m framework.compare_backtests --old output --new output --export-md --out-dir output

# Restrict to a specific model id (repeatable or comma-separated)
python -m framework.compare_backtests --old output --new output --model hybrid --export-md

# Show only meaningful changes (absolute delta >= tolerance)
python -m framework.compare_backtests --old output --new output --only-changed --tolerance 0.001 --export-md
```

Outputs:
- backtest_compare_<timestamp>_deltas.(csv|md): columns include model_id, train_window, test_window, and delta_* fields for ECE, Brier, NLL, baseline NLL, and Skill vs Baseline.
- backtest_compare_<timestamp>_summary.(csv|md): mean/median deltas across all compared models.

Notes and diagnostics:
- Keys used for matching rows: (model_id, train_window, test_window). Both files must contain matching rows for deltas to compute.
- The CLI prints the resolved file paths (OLD/NEW) so you can see what’s being compared.
- If there’s no overlap, it prints available windows and models in each file with a tip to align them.
- If you pass the same directory for both --old and --new, the tool auto-picks the previous file as OLD and the latest as NEW.
- Use --model to focus on shared models, and --only-changed/--tolerance to highlight significant differences.

### Execution Parameters

**Primary Forecast Script** (`forecast.py`):
```bash
python forecast.py
```

**Environment Variables**:
- `INCIDENTS_CSV`: Path to incident data file (optional)
- `EXPORT_MD`: Enable/disable Markdown report generation (default: true)
- `EXPORT_JSON`: Enable/disable JSON report generation (default: true)
- `REPORT_BASE_NAME`: Base filename for reports (default: clustercast_report)
- `SPLIT_OUTPUTS`: Split outputs into predictions/calibration (md/json) (default: true). Set to `false` to produce a single legacy combined report.
- `APPLY_CALIBRATION`: Apply post-calibration using bins if available (default: true)
- `REBUILD_CALIBRATION`: Rebuild calibration artifacts before each forecast (default: true)
 - `CALIBRATION_N_BINS`: Number of calibration bins (quantile-based; default: 20)
 - `CAL_MIN_COUNT_PER_BIN`: Minimum samples per bin for adaptive merging (default: 100)

**Backtest Module** (`self_optimizing_backtest.py`):
```python
from self_optimizing_backtest import do_backtest

# Run all models with expanded parameter ranges
results = do_backtest(
    model_name='all',  # Options: 'baseline', 'contagion', 'hybrid', 'all'
    data_file='data/incidents.csv',
    decay_values=np.arange(0.1, 1.0, 0.05),  # DECAY parameter range
    jump_values=np.arange(0.001, 0.201, 0.01)  # JUMP parameter range
)
```

### Data Requirements

**Input Data Format** (CSV):
```csv
Company/Domain Name,Group Name,Discovered Date,Sector,Bundesland
Deutsche Bahn,wannacry,2017-05-12,transportation,Berlin
KrausMaffei,bitpaymer,2018-11-21,manufacturing,Bavaria (Bayern)
```

**Required Columns**:
- `Discovered Date`: Incident discovery timestamp (YYYY-MM-DD format)
- `Bundesland`: German federal state designation
- `Sector`: Economic sector classification (must be a STIX 2.1 Industry Sector Open Vocabulary Name (industry-sector-ov) )

**Data Location**: The system looks for incident data in the following order:
1. Path specified by `INCIDENTS_CSV` environment variable
2. `data/incidents.csv` relative to the script directory (recommended location)
3. `incidents.csv` in the script directory
4. `incidents.csv` in common locations (./data, ../data)

**Vocabulary Summary**

>agriculture, aerospace, automotive, chemical, commercial, communications, construction, defense, education, energy, entertainment, financial-services, government (emergency-services, government-local, government-national, government-public-services,  government-regional), healthcare, hospitality-leisure, infrastructure (dams, nuclear, water), insurance, manufacturing, mining, non-profit, pharmaceuticals, retail, technology, telecommunications, transportation, utilities

**Data Preprocessing**:
- Automatic encoding handling (UTF-8 with Latin-1 fallback)
 - Missing sector values defaulted to "other" and normalized (dash characters unified)
- Weekly aggregation by Bundesland-Sector combinations
- Temporal indexing for rolling window operations

### Output Specifications

**JSON Report Structure**:
```json
{
  "report_timestamp": "2025-10-04 11:15:24",
  "model_forecasts": [
    {
      "model_config": "hybrid (26/13 Opt)",
      "optimal_decay": 0.95,
      "optimal_jump": 0.19,
      "filtered_targets": [
        {
          "Target": "Bavaria (Bayern) | manufacturing",
          "Average Weekly Probability": 0.4491,
          "Risk Band": "Very High"
        }
      ]
    }
  ]
}
```

**Markdown Report Features**:
- Risk band interpretations with actionable recommendations
- Sorted target lists by probability (descending)
- Combined risk aggregations by Bundesland and Sector
- Statistical summary tables with formatting

### JSON fields and deprecations (calibration-awareness)

In split-output mode (default):
- `predictions.json` focuses on predictions. It includes minimal calibration status fields (`calibration_applied`, `calibration_method`, `calibration_param`, `calibration_smoothing_alpha`, `calibration_source`) only as provenance on how predictions were produced, and omits calibration metrics.
- `calibration.json` contains calibration metrics and metadata for each model/horizon.

Deprecated in `predictions.json` (split mode):
- The `calibration_metrics` object (ECE, Brier, NLL) is deprecated from `predictions.json` and will not be populated. Use `calibration.json` instead.
- The root-level `calibration` section is also deprecated from `predictions.json`; use the standalone `calibration.json`.

In legacy combined mode (`SPLIT_OUTPUTS=false`), `predictions.json` may still include a root-level `calibration` section and per-model `calibration_metrics` for backward compatibility. This path will be removed in a future major release.

## Dependencies and Technical Requirements

**Core Libraries**:
- `pandas` ≥ 1.5.0: Data manipulation and time series handling
- `numpy` ≥ 1.21.0: Numerical computations and array operations
- `json`: Report serialization (standard library)
- `os`, `glob`: File system operations (standard library)

**Python Version**: 3.8+ (type hints and advanced features)

**System Requirements**:
- Memory: Minimum 4GB RAM for optimization runs
- Storage: 100MB for intermediate calculations and reports
- Processing: Multi-core CPU recommended for parameter optimization

**No External API Dependencies**: System operates entirely on local data and computations for security and reliability.

## Calibration

### Purpose
Model probabilities can be accurate but miscalibrated (over- or under-confident). We add a lightweight post-calibration step to better align predicted probabilities with observed frequencies while preserving relative ranking.

### Method: Adaptive Histogram Binning (Out-of-sample)
 - We compute reliability curves on backtest predictions and store per-bin statistics under `output/calibration_assets` using `run_calibration_analysis()` from `self_optimizing_backtest.py`.
 - Bins are derived from quantiles and merged until each has at least `CAL_MIN_COUNT_PER_BIN` samples, stabilizing sparse tails.
 - During daily forecasts, we optionally map raw probabilities to the observed event rate of the nearest bin center (histogram binning). This is a simple, robust, non-parametric post-calibration.

Artifacts (auto-generated by `run_calibration_analysis`):
 - `output/calibration_assets/calibration_summary.json` - ECE, Brier, NLL per model/horizon
 - `output/calibration_assets/reliability_<model>.png` - reliability diagrams with raw+calibrated overlay, Wilson 95% CIs, and per-bin counts
 - `output/calibration_assets/histogram_<model>.png` - probability histograms (raw and calibrated)
 - `output/calibration_assets/bins_<model>.csv` - columns: `bin_center`, `bin_count`, `mean_pred`, `event_rate`, `event_rate_lo95`, `event_rate_hi95`
 - `output/calibration_assets/segment_breakdown.csv` - ECE/Brier per Bundesland and Sector

Toggle calibration at forecast time via environment variable:
- `APPLY_CALIBRATION=true` (default) - apply histogram-binning calibration if bins are available
- `APPLY_CALIBRATION=false` - skip post-calibration (use raw model probabilities)

Automatic rebuild: By default, the forecast run triggers a calibration rebuild (backtests → reliability curves → updated `bins_*.csv`) before computing probabilities. This ensures the forecast never uses stale calibration. Control this with `REBUILD_CALIBRATION`.

In split-output mode (default):
- The predictions report (predictions.md/json) focuses on predictions and a concise status showing whether calibration was applied (e.g., “Calibrated (histogram_binning)”).
- The calibration report (calibration.md/json) contains the calibration summary (ECE/Brier/NLL), plots, histograms, CI tables, per-segment snapshots, and method/parameter details.

### Per-model calibration method selection (automatic)

 - During calibration rebuilds, the system compares multiple calibrators per model/horizon:
   - Histogram binning (with optional Laplace smoothing)
   - Isotonic regression (PAV)
   - Temperature scaling (logit rescaling by 1/T)
   - Intensity scaling (Poisson intensity λ scaled by factor s)
   Selection is based primarily on ECE (tie-break by Brier), with grid search for T or s where applicable.

  Results are written to `output/calibration_assets/calibration_method_comparison.json`, including per-model recommendations and parameters.

   The forecast pipeline reads this artifact and applies the per-model recommendation by default. If missing or unreadable, it falls back to histogram binning (with smoothing if enabled).

Override behavior via environment variables:
 - `CALIBRATION_METHOD`: global override for all models - `histogram` | `isotonic` | `temperature` | `intensity`
   - When unset or empty, per-model recommendations are used.
 - `CAL_APPLY_SMOOTHING` and `CAL_LAPLACE_ALPHA` apply when using bin-based methods; event-rate targets are smoothed before mapping.
 
Note: If overriding to `temperature` or `intensity` and no parameter is available from the comparison artifact, neutral defaults are used (`T=1.0`, `s=1.0`).

Note: The calibration rebuild step generates both the usual reliability artifacts and the comparison artifact automatically, so recommendations are kept fresh.

### How to generate/update calibration artifacts
Run the calibration analysis after any meaningful model/data update to refresh the bins and plots:

```powershell
python -c "from self_optimizing_backtest import run_calibration_analysis, run_calibration_method_comparison; ^
run_calibration_analysis('all','data/incidents.csv', n_bins=20, min_count_per_bin=100, output_dir='output/calibration_assets'); ^
run_calibration_method_comparison('all','data/incidents.csv', n_bins=20, laplace_alpha=0.5, min_count_per_bin=100, output_dir='output/calibration_assets')"
```

This writes the summary JSON (with NLL), per-bin CSVs (with 95% CIs), reliability plots (raw+calibrated overlays with CIs and counts), probability histograms, the method-comparison artifact (with per-model recommendations and parameters), and the per-segment breakdown. The forecast then picks up the latest artifacts automatically when `APPLY_CALIBRATION` is enabled.

### Notes and alternatives
- Histogram binning is intentionally simple and robust for sparse data. For stricter calibration with minimal loss of sharpness, consider temperature scaling or isotonic regression per horizon.

## Validation and Quality Assurance

**Model Validation Protocol**:
1. **Data Integrity Checks**: Automatic validation of required columns and data types
2. **Temporal Consistency**: Chronological ordering verification and gap detection
3. **Statistical Validation**: Range checks for probability outputs and parameter bounds
4. **Backtesting Framework**: Historical performance validation across multiple time periods

**Performance Benchmarks**:
- **Baseline Comparison**: All advanced models measured against historical frequency
- **Skill Score Targets**: Minimum 5% improvement over baseline for deployment
- **Computational Efficiency**: Sub-minute execution for routine forecasting operations

**Error Handling and Robustness**:
- **Graceful Data File Resolution**: Multiple fallback strategies for data location
- **Encoding Tolerance**: Automatic handling of UTF-8 and Latin-1 character encodings
- **Missing Data Management**: Default values and filtering strategies for incomplete records
- **Parameter Boundary Enforcement**: Automatic clipping and validation of optimization results

### Output validators (split reports)

To sanity-check report artifacts after a run, use the lightweight validators:

- Structure check: asserts split-mode outputs exist (predictions/calibration in md/json) and `output/calibration_assets/` is present.
- Link check: verifies all image/CSV links embedded in `output/calibration.md` resolve.

Run the validator CLI from the `src` directory:

```
python -m framework.validate_cli
```

Result prints PASS/FAIL with specific issues (missing files or broken links).

## JSON Report Fields (Calibration-aware)

Each `model_forecasts[]` entry now includes:
- `calibration_applied` (bool)
 - `calibration_method` (string; `histogram_binning` | `isotonic` | `temperature` | `intensity`)
 - `calibration_param` (number; `T` for temperature or `s` for intensity when applicable)
 - `calibration_smoothing_alpha` (number; when bin-based methods are used with smoothing)
 - `calibration_source` (path to bins CSV for bin-based methods, or the method-comparison JSON for temperature/intensity)
 - `calibration_metrics` (object; optional) with `ece`, `brier`, `nll`, `test_window_weeks` when `calibration_summary.json` is available

Root-level `calibration` section:
- `artifacts_dir`: path to the calibration artifacts directory
- `generated_at`: timestamp derived from `calibration_summary.json`'s modified time
 - `summary`: array of `{ model_config, test_window_weeks, ece, brier, nll }` from `calibration_summary.json`

This comprehensive framework provides enterprise-grade ransomware risk forecasting with mathematical rigor, operational interpretability, and technical robustness suitable for cybersecurity decision-making in complex organizational environments.

## Calibration options and robustness

In addition to default histogram-binning calibration, the forecast pipeline supports multiple robust calibrators:

1) Histogram bin smoothing (default on)
- Adds Laplace (additive) smoothing to bin event rates to reduce extreme 0/1 mappings when bins are sparse.
- Environment variables:
  - `CAL_APPLY_SMOOTHING=true|false` (default `true`)
  - `CAL_LAPLACE_ALPHA=<float>` (default `0.5`)

2) Isotonic regression calibrator (optional)
- Fits a non-decreasing mapping from predicted probabilities to observed event rates using the PAV algorithm over binned points (weighted by bin counts).
- Environment variable:
  - `CALIBRATION_METHOD=histogram|isotonic|temperature|intensity` (default `histogram` unless per-model selection available)
- On failure during fitting, the pipeline falls back to histogram binning automatically.

3) Temperature scaling (optional)
- Applies p' = sigmoid(logit(p)/T). T>1 reduces confidence; T<1 increases confidence.

4) Intensity scaling (optional)
- For Poisson-based probabilities: λ' = s·λ, so p' = 1 - exp(-λ'). s>1 increases intensity; s<1 decreases it.

How to switch calibrators globally (PowerShell examples):

```powershell
# Histogram + smoothing (default)
$env:REBUILD_CALIBRATION = 'false'
$env:APPLY_CALIBRATION = 'true'
$env:CAL_APPLY_SMOOTHING = 'true'
$env:CAL_LAPLACE_ALPHA = '0.5'
$env:CALIBRATION_METHOD = 'histogram'
python path\src\forecast.py

# Isotonic
$env:REBUILD_CALIBRATION = 'false'
$env:APPLY_CALIBRATION = 'true'
$env:CAL_APPLY_SMOOTHING = 'true'
$env:CAL_LAPLACE_ALPHA = '0.5'
$env:CALIBRATION_METHOD = 'isotonic'
python path\src\forecast.py

# Temperature scaling (uses per-model T if available, else T=1.0)
$env:REBUILD_CALIBRATION = 'false'
$env:APPLY_CALIBRATION = 'true'
$env:CALIBRATION_METHOD = 'temperature'
python path\src\forecast.py

# Intensity scaling (uses per-model s if available, else s=1.0)
$env:REBUILD_CALIBRATION = 'false'
$env:APPLY_CALIBRATION = 'true'
$env:CALIBRATION_METHOD = 'intensity'
python path\src\forecast.py
```

Markdown and JSON outputs (split mode default)
- predictions.md: presentation-focused predictions, executive summary, filtered target tables, combined risk overviews, risk band interpretations, and a concise “Calibrated (method)” status per model.
- calibration.md: calibration summary tables (ECE/Brier/NLL), a Summary Metrics (Raw vs Calibrated) table with deltas (ΔECE/ΔBrier/ΔNLL), reliability plots and probability histograms, bin CI tables, per-segment snapshots, and methods/parameters with sources.
- predictions.json: model forecasts, target probabilities, and minimal calibration status fields.
- calibration.json: artifacts directory and per-model/horizon metrics; suitable for programmatic inspection.

Legacy combined report
- You can temporarily revert to the legacy single combined report by setting `SPLIT_OUTPUTS=false`. A read fallback to `output/calibration/` exists during the transition and may be removed later.
