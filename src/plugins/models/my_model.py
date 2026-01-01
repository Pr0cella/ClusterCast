from __future__ import annotations

from typing import Any, List
import numpy as np
from framework.base import ModelInterface, SearchParam


class MyModel(ModelInterface):
    id = 'my_model'
    name = 'My Model'
    version = '0.1.0'

    def required_features(self) -> List[str]:
        return ['week', 'Target_V']

    def get_search_space(self):
        return [
            SearchParam('param', kind='float', low=0.0, high=1.0, step=0.1),
        ]

    def fit(self, train_df):
        # Example: baseline probability per target as state
        lam = train_df.mean(axis=0).values
        p = 1 - np.exp(-lam)
        return {'p_base': p}

    def predict_proba(self, state: Any, context: dict) -> np.ndarray:
        # Example: use baseline state (and optional param) to produce probabilities
        param = float(self.get_params().get('param', 0.0))
        p = np.clip(state['p_base'] + param, 0.0, 1.0)
        return p
