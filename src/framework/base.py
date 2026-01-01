from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence, Callable
import numpy as np


@dataclass
class SearchParam:
    name: str
    kind: str  # 'float' | 'int' | 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    values: Optional[Sequence[Any]] = None


@dataclass
class EvaluationMetrics:
    ece: Optional[float] = None
    brier: Optional[float] = None
    nll: Optional[float] = None
    skill_vs_baseline: Optional[float] = None


@dataclass
class Artifacts:
    run_id: str
    output_dir: str
    predictions_json: Optional[str] = None
    predictions_md: Optional[str] = None
    calibration_json: Optional[str] = None
    calibration_md: Optional[str] = None
    calibration_assets_dir: Optional[str] = None


class ModelInterface(ABC):
    id: str
    name: str
    version: str

    def __init__(self, horizon_weeks: int = 1):
        self._horizon_weeks = horizon_weeks
        self._params: Dict[str, Any] = {}

    @property
    def horizon_weeks(self) -> int:
        return self._horizon_weeks

    def set_params(self, **params):
        self._params.update(params)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._params)

    @abstractmethod
    def required_features(self) -> List[str]:
        ...

    @abstractmethod
    def get_search_space(self) -> List[SearchParam]:
        ...

    @abstractmethod
    def fit(self, train_df) -> Any:
        """Return a model state to be used in predict/update_state."""
        ...

    @abstractmethod
    def predict_proba(self, state: Any, context: Dict[str, Any]) -> np.ndarray:
        ...

    def update_state(self, state: Any, observed_counts: np.ndarray) -> Any:
        """Optional state update given observed counts. Default no-op."""
        return state


class CalibratorInterface(ABC):
    method_name: str

    @abstractmethod
    def fit(self, preds: np.ndarray, labels: np.ndarray, **context) -> Any:
        ...

    @abstractmethod
    def apply(self, preds: np.ndarray, state: Any) -> np.ndarray:
        ...


class OptimizerInterface(ABC):
    @abstractmethod
    def optimize(self,
                 model: ModelInterface,
                 data,
                 objective: Callable[[Dict[str, Any]], float],
                 budget: int = 50) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Return best_params, trials list with {'params','score'}"""
        ...


class EvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(self, preds: np.ndarray, labels: np.ndarray) -> EvaluationMetrics:
        ...


class ReporterInterface(ABC):
    @abstractmethod
    def render_predictions(self, artifacts: Artifacts, **kwargs) -> None:
        ...

    @abstractmethod
    def render_calibration(self, artifacts: Artifacts, **kwargs) -> None:
        ...
