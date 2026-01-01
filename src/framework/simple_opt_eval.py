from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Tuple, Callable
import numpy as np
from .base import OptimizerInterface, EvaluatorInterface, EvaluationMetrics, SearchParam


class RandomSearchOptimizer(OptimizerInterface):
    def optimize(self,
                 model,
                 data,
                 objective: Callable[[Dict[str, Any]], float],
                 budget: int = 50) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        space: List[SearchParam] = model.get_search_space()
        def sample_param(p: SearchParam):
            if p.kind == 'float':
                if p.step:
                    n = int(math.floor((p.high - p.low) / p.step))
                    idx = random.randint(0, max(0, n))
                    return p.low + idx * p.step
                return random.uniform(p.low, p.high)
            if p.kind == 'int':
                return random.randint(int(p.low), int(p.high))
            if p.kind == 'categorical':
                return random.choice(list(p.values or []))
            raise ValueError(f"Unknown param kind: {p.kind}")

        trials: List[Dict[str, Any]] = []
        best = None
        best_score = float('inf')
        for _ in range(budget):
            params = {p.name: sample_param(p) for p in space}
            score = objective(params)
            trials.append({'params': params, 'score': score})
            if score < best_score:
                best_score = score
                best = params
        return best or {}, trials


class SimpleEvaluator(EvaluatorInterface):
    def evaluate(self, preds: np.ndarray, labels: np.ndarray) -> EvaluationMetrics:
        eps = 1e-15
        p = np.clip(preds.astype(float), eps, 1 - eps).ravel()
        y = labels.astype(float).ravel()
        # NLL
        nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        # Brier
        brier = float(np.mean((p - y) ** 2))
        # ECE (fixed 10 bins)
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        ece = 0.0
        for b in range(10):
            mask = idx == b
            if np.any(mask):
                pb = float(np.mean(p[mask]))
                ob = float(np.mean(y[mask]))
                w = float(np.mean(mask))
                ece += w * abs(pb - ob)
        return EvaluationMetrics(ece=ece, brier=brier, nll=nll)
