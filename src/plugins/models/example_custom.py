from __future__ import annotations

from typing import Any, List
import numpy as np
from framework.base import ModelInterface, SearchParam


class ExampleCustomModel(ModelInterface):
    id = 'custom_example'
    name = 'Custom Example Model'
    version = '0.1.0'

    def required_features(self) -> List[str]:
        return ['week', 'Target_V']

    def get_search_space(self):
        return [
            SearchParam('bias', kind='float', low=-0.05, high=0.05, step=0.01),
        ]

    def fit(self, train_df):
        # Store a simple baseline probability per target
        lam = train_df.mean(axis=0).values
        p = 1 - np.exp(-lam)
        return {'p_base': p}

    def predict_proba(self, state: Any, context: dict) -> np.ndarray:
        bias = float(self.get_params().get('bias', 0.0))
        p = state['p_base'] + bias
        return np.clip(p, 0.0, 1.0)
