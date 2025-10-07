# Changelog

All notable changes to this project will be documented in this file.

## [2025-10-05]

### Added
- Split outputs are now the default: `output/predictions.(md|json)` and `output/calibration.(md|json)` with artifacts under `output/calibration_assets/`.
- Calibration report (`calibration.md`) includes a "Summary Metrics (Raw vs Calibrated)" table with deltas (ΔECE/ΔBrier/ΔNLL), plus reliability plots, histograms, and bin CI tables.
- Validators: `framework/validate_cli.py` to check structure and link integrity.
- Modular framework scaffolding under `framework/` (interfaces, adapters, optimizer/evaluator, reporter, registry).
- Modular runner: `python -m framework.modular_runner` to run built-in or plugin models; flags for optimization, plugins directory, listing models, and emitting reports.
- Reporter: `framework/reporter.py` that writes split reports using existing `forecast.py` generators.
- Plugin discovery: `framework/plugin_loader.py` loads `ModelInterface` subclasses from `src/plugins/models/`.
- Example plugin: `src/plugins/models/example_custom.py` (id: `custom_example`).
- Plugin scaffold CLI: `python -m framework.scaffold_plugin` to generate a new plugin model file.
- Modular backtest CLI: `python -m framework.modular_backtest` to run rolling backtests for built-in and plugin models with baseline comparison.
	- Supports multiple `--model` values and `--all-models`.
	- Export results to CSV/Markdown with `--export-csv` / `--export-md` and `--out-dir`.
	- Backtest compare CLI: `python -m framework.compare_backtests` to diff two backtest CSVs (per-model deltas and aggregate summary) with CSV/Markdown export options.

### Changed
- `forecast.py` execution wrapped under `if __name__ == "__main__":` to make its helpers safe to import without running heavy report generation.
- Paths in reports normalized to forward slashes for Markdown portability.

### Deprecated
- In split mode, `predictions.json` no longer carries calibration metrics (`calibration_metrics`) or a root-level `calibration` section. Use `calibration.json` instead. Legacy combined mode still includes them but is slated for removal in a future major release.

### Notes
- A temporary read-fallback to `output/calibration/` exists during transition.
