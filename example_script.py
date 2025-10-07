from framework.adapters import HybridAdapter
from framework.simple_opt_eval import RandomSearchOptimizer
from forecast import prepare_data, calculate_log_loss
import numpy as np
import pandas as pd

# Load enough history (train + horizon)
Y_t, n_targets, total_weeks, _ = prepare_data(None, required_history_weeks=26, forecast_weeks=13)

model = HybridAdapter(horizon_weeks=13)

# Optional: tiny optimization on last horizon using NLL
optimizer = RandomSearchOptimizer()
def objective(params):
    model.set_params(**params)
    start = total_weeks - 13
    state = model.fit(Y_t.iloc[start - 26 : start])
    nlls = []
    for step in range(13):
        t = start + step
        ctx = {'Y_train': Y_t.iloc[t - 26 : t], 'Y_current': Y_t.iloc[t].values}
        p = model.predict_proba(state, ctx)
        y = (Y_t.iloc[t].values > 0).astype(int)
        nlls.append(calculate_log_loss(y, p))
    return float(np.mean(nlls))

best, _ = optimizer.optimize(model, None, objective, budget=5)
model.set_params(**best)

# Fit once and forecast avg weekly probabilities
start = total_weeks - 13
state = model.fit(Y_t.iloc[start - 26 : start])
preds = []
for step in range(13):
    t = start + step
    ctx = {'Y_train': Y_t.iloc[t - 26 : t], 'Y_current': Y_t.iloc[t].values}
    preds.append(model.predict_proba(state, ctx))
avg_weekly = np.mean(np.array(preds), axis=0)

s = pd.Series(avg_weekly, index=Y_t.columns)
print(s.sort_values(ascending=False).head(10))