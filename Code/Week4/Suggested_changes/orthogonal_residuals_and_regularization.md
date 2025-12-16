# Suggested Changes: Handling Wind/Pressure Correlation Without Losing Interpretability

This note proposes changes and options to reduce multicollinearity between wind and pressure while preserving interpretability and predictive power in your current model.

Files referenced: [Speciale/Code/Week4/Bayes_week4.py](Speciale/Code/Week4/Bayes_week4.py)

---

## 1) Orthogonal Residualization (FWL-style) — Recommended Short-Term

Goal: keep a physics-based wind term and add an independent “extra wind” component that is orthogonal to it.

Why: including `wind_obs`, `modelled_wind`, and a naive `wind_residual = wind_obs - modelled_wind` together creates multicollinearity and unstable coefficients. Residualizing `wind_obs` on `modelled_wind` removes the shared signal.

Implementation sketch:

```python
from sklearn.linear_model import LinearRegression

# After computing modelled_wind and wind_speed (observed)
lr = LinearRegression().fit(modelled_wind.reshape(-1, 1), wind_speed)
wind_pred = lr.predict(modelled_wind.reshape(-1, 1))
residual_wind = wind_speed - wind_pred  # orthogonal to modelled_wind

# In the PyMC model, include either:
#   - modelled_wind + residual_wind (interpretable physics + departures)
# or
#   - wind_obs + pressure_resid (see Section 2)

if model_spec.get("modelled_wind", False):
    modelled_wind_coef = pm.Normal("modelled_wind_coef", sigma=3)
    mu = mu + modelled_wind_coef * modelled_wind

if model_spec.get("wind_residual", False):
    wind_resid_coef = pm.Normal("wind_resid_coef", sigma=3)
    mu = mu + wind_resid_coef * residual_wind
```

Interpretation:
- `modelled_wind_coef`: effect via the pressure→wind physics mapping.
- `wind_resid_coef`: effect of wind departures not captured by the physics (terrain, structure, sampling, etc.).

Do not include all three of `wind_obs`, `modelled_wind`, and `wind_residual` simultaneously. Use one raw path plus one residualized path.

---

## 2) Symmetric Variant (Wind-centric): `wind_obs` + `pressure_resid`

If you want a wind-centered story, residualize pressure on wind and include the raw wind term.

```python
# pressure_resid = pressure − E[pressure | wind]
lr_pw = LinearRegression().fit(wind_speed.reshape(-1,1), pressure)
pressure_pred = lr_pw.predict(wind_speed.reshape(-1,1))
pressure_resid = pressure - pressure_pred

# Model terms
if model_spec.get("wind_obs", False):
    wind_speed_coef_obs = pm.Normal("wind_speed_coef_obs", sigma=3)
    mu = mu + wind_speed_coef_obs * wind_speed

if model_spec.get("pressure_resid", False):
    pressure_resid_coef = pm.Normal("pressure_resid_coef", sigma=3)
    mu = mu + pressure_resid_coef * pressure_resid
```

This reads as “effect of wind + any barometric effect not already explained by wind.”

---

## 3) Latent Intensity Factor — Principled and Interpretable

Introduce a latent storm intensity `S_i`. Wind and pressure are conditionally independent measurements of `S_i`; damage depends on `S_i`.

```python
with pm.Model() as m:
    S = pm.Normal("S", 0, 1, shape=len(observed))  # latent intensity

    # Measurement models (calibrate; signs/links can be adapted)
    c_w  = pm.Normal("c_w", 0, 5)
    b_w  = pm.HalfNormal("b_w", 2)
    tau_w = pm.HalfNormal("tau_w", 2)
    pm.Normal("wind_like", mu=c_w + b_w*S, sigma=tau_w, observed=wind_speed)

    c_p  = pm.Normal("c_p", 0, 5)
    b_p  = pm.HalfNormal("b_p", 2)
    tau_p = pm.HalfNormal("tau_p", 2)
    pm.Normal("pressure_like", mu=c_p - b_p*S, sigma=tau_p, observed=pressure)

    # Damage model
    alpha = pm.Normal("alpha", 17, 5)
    gamma = pm.Normal("gamma", 0, 3)
    sigma = pm.HalfNormal("sigma", 5)

    mu = alpha + gamma*S
    # Add economic/tide/travel_speed/trend terms as in your current model
    pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
```

Interpretation remains clean: `gamma` is the damage sensitivity to intensity; `b_w, b_p` connect measurements to intensity.

---

## 4) Regularization Options (if you keep both raw predictors)

If you want both raw wind and pressure in the same linear predictor, stabilize estimates with shrinkage.

### 4.a) Ridge-like priors

```python
pressure_coef = pm.Normal("pressure_coef", mu=0, sigma=1.0)
wind_speed_coef_obs = pm.Normal("wind_speed_coef_obs", mu=0, sigma=1.0)
```

Smaller `sigma` reduces variance under multicollinearity.

### 4.b) (Regularized) Horseshoe prior

```python
# Global shrinkage
tau = pm.HalfCauchy("tau", beta=1)

# Local shrinkage for each coefficient in the wind/pressure block
lambda_w = pm.HalfCauchy("lambda_w", beta=1)
lambda_p = pm.HalfCauchy("lambda_p", beta=1)

# Coefficients
wind_speed_coef_obs = pm.Normal("wind_speed_coef_obs", mu=0, sigma=tau*lambda_w)
pressure_coef = pm.Normal("pressure_coef", mu=0, sigma=tau*lambda_p)
```

For better sampling, use a regularized horseshoe (e.g., replace HalfCauchy with HalfStudentT or add slab variance). This keeps both features, shrinks aggressively when redundant, and lets important effects escape.

---

## 5) Quick Diagnostics (before modeling)

- Pairwise correlation of active predictors; if `|ρ| > 0.9`, prefer residualization or the latent factor.
- VIFs (>10 suggests severe multicollinearity). Optional snippet:

```python
import numpy as np
import pandas as pd

X = pd.DataFrame({
    k: v for k, v in {
        'pressure': pressure,
        'wind_speed': wind_speed,
        'modelled_wind': modelled_wind,
        'residual_wind': residual_wind,
    }.items() if k in active_features  # build from model_spec
})
print("\nPairwise correlation:\n", X.corr())

# Simple VIF without extra deps (manual):
# VIF_j = 1 / (1 - R^2_j), where R^2_j is from regressing x_j on X_-j
from sklearn.linear_model import LinearRegression
for col in X.columns:
    y = X[col].values
    Z = X.drop(columns=[col]).values
    r2 = LinearRegression().fit(Z, y).score(Z, y)
    vif = 1.0 / max(1e-8, (1 - r2))
    print(f"VIF({col}) = {vif:.2f}")
```

---

## 6) Small Spec Key Fix

Ensure the spec key is `"modelled_wind"` (not `"wind_modelled"`) so the modeled wind term is actually included.

Example specs:

```python
model_specs = [
    {"economic": True, "tides": True, "wind_obs": True, "wind_residual": True, "travel_speed": True, "trend": True},
    {"economic": True, "pressure": True, "tides": True, "wind_obs": True, "travel_speed": True, "trend": True},
    {"economic": True, "pressure": True, "tides": True, "modelled_wind": True, "wind_residual": True, "travel_speed": True, "trend": True},
]
```

---

## 7) Practical Guidance

- For interpretable inference now: use `modelled_wind` + orthogonal `wind_residual` (or the symmetric `wind_obs` + `pressure_resid`).
- For a principled causal structure: move to the latent intensity model.
- If you must keep both raw predictors: add stronger shrinkage (ridge or horseshoe) to stabilize attribution.
- Avoid including `wind_obs`, `modelled_wind`, and `wind_residual` all together.

If you want, I can wire any option above as a toggle in [Speciale/Code/Week4/Bayes_week4.py](Speciale/Code/Week4/Bayes_week4.py) without changing your default path.
