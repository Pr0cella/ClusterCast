from __future__ import annotations

from typing import Any, List
import numpy as np
from .base import ModelInterface, SearchParam

# Support running both as a package module and from src
try:
    # When framework is a package and src is a package parent (rare)
    from ..forecast import baseline_forecast, contagion_only_forecast, hybrid_forecast  # type: ignore
except Exception:
    # Default: import from script directory on sys.path
    from forecast import baseline_forecast, contagion_only_forecast, hybrid_forecast  # type: ignore


class BaselineAdapter(ModelInterface):
    id = 'baseline'
    name = 'Historical Frequency Baseline'
    version = '1.0.0'

    def required_features(self) -> List[str]:
        return ['week', 'Target_V']

    def get_search_space(self):
        return []

    def fit(self, train_df):
        return {'train_df': train_df}

    def predict_proba(self, state: Any, context: dict) -> np.ndarray:
        Y_train = context.get('Y_train')
        out = baseline_forecast(Y_train)
        return out.values


class ContagionAdapter(ModelInterface):
    id = 'contagion'
    name = 'Contagion-Only'
    version = '1.0.0'

    def required_features(self) -> List[str]:
        return ['week', 'Target_V']

    def get_search_space(self):
        return [
            SearchParam('decay', kind='float', low=0.1, high=0.99, step=0.05),
            SearchParam('jump', kind='float', low=0.001, high=0.2, step=0.01),
        ]

    def fit(self, train_df):
        # Initialize Hawkes memory as zeros
        n_targets = train_df.shape[1]
        return {'H_prev': np.zeros(n_targets)}

    def predict_proba(self, state: Any, context: dict) -> np.ndarray:
        Y_train = context.get('Y_train')
        Y_current = context.get('Y_current')
        decay = float(self.get_params().get('decay', 0.9))
        jump = float(self.get_params().get('jump', 0.05))
        p, H = contagion_only_forecast(Y_train, Y_current, state['H_prev'], decay, jump)
        state['H_prev'] = H
        return p


class HybridAdapter(ModelInterface):
    id = 'hybrid'
    name = 'Hybrid'
    version = '1.0.0'

    def required_features(self) -> List[str]:
        return ['week', 'Target_V']

    def get_search_space(self):
        return [
            SearchParam('decay', kind='float', low=0.1, high=0.99, step=0.05),
            SearchParam('jump', kind='float', low=0.001, high=0.2, step=0.01),
        ]

    def fit(self, train_df):
        n_targets = train_df.shape[1]
        return {'H_prev': np.zeros(n_targets)}

    def predict_proba(self, state: Any, context: dict) -> np.ndarray:
        Y_train = context.get('Y_train')
        Y_current = context.get('Y_current')
        decay = float(self.get_params().get('decay', 0.9))
        jump = float(self.get_params().get('jump', 0.05))
        p, H = hybrid_forecast(Y_train, Y_current, state['H_prev'], decay, jump)
        state['H_prev'] = H
        return p
