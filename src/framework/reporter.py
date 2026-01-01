from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import pandas as pd

from .base import ReporterInterface, Artifacts

# Import report generators and thresholds from forecast module
import forecast as fc


class SplitReporter(ReporterInterface):
    """Reporter that writes the split predictions/calibration reports using forecast.py helpers."""

    def __init__(self, output_dir: Optional[str] = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self._default_output = os.path.normpath(os.path.join(script_dir, '..', 'output'))
        self._output_dir = output_dir or self._default_output
        os.makedirs(self._output_dir, exist_ok=True)

    def render_predictions(self, artifacts: Artifacts, **kwargs) -> None:
        forecast_results: List[Dict[str, Any]] = kwargs.get('forecast_results', [])
        export_md: bool = bool(kwargs.get('export_md', True))
        export_json: bool = bool(kwargs.get('export_json', True))
        predictions_base: str = kwargs.get('predictions_base', 'predictions')

        # Write JSON/MD using forecast.generate_risk_report (predictions-only)
        if export_json:
            json_path = os.path.join(self._output_dir, f"{predictions_base}.json")
            fc.generate_risk_report(
                forecast_results,
                fc.MIN_AVG_WEEKLY_PROBABILITY,
                fc.PROBABILITY_BANDS,
                output_format='json',
                output_file=json_path,
                include_calibration_sections=False,
            )
            artifacts.predictions_json = json_path

        if export_md:
            md_path = os.path.join(self._output_dir, f"{predictions_base}.md")
            fc.generate_risk_report(
                forecast_results,
                fc.MIN_AVG_WEEKLY_PROBABILITY,
                fc.PROBABILITY_BANDS,
                output_format='markdown',
                output_file=md_path,
                include_calibration_sections=False,
            )
            artifacts.predictions_md = md_path

    def render_calibration(self, artifacts: Artifacts, **kwargs) -> None:
        export_md: bool = bool(kwargs.get('export_md', True))
        export_json: bool = bool(kwargs.get('export_json', True))
        if export_json:
            cjson = os.path.join(self._output_dir, 'calibration.json')
            fc.generate_calibration_report(output_format='json', output_file=cjson)
            artifacts.calibration_json = cjson
        if export_md:
            cmd = os.path.join(self._output_dir, 'calibration.md')
            fc.generate_calibration_report(output_format='markdown', output_file=cmd)
            artifacts.calibration_md = cmd
