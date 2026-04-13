# Model Naming Convention with Likelihood Suffixes

## Likelihood Abbreviations

| Likelihood | Abbreviation | Meaning |
|-----------|--------------|---------|
| Log-Normal | `ln` | Log-normal (standard, default) |
| Log-Student's t | `lst` | Log-Student's t (heavy tails) |
| Pareto | `pw` | Pareto/power-law |
| ULLN | `ulln` | Upper Limit Log-Normal |

## Model Name Structure

```
{observed_variable}_hurricane_model_{features}_{likelihood}
```

### Examples

**Exposure + Pressure + Trend + Surge + Alpha with log-normal:**
```
basedamage_hurricane_model_ec_p_tr_surg_f_alpha_ln
                                                  ^^
```

**Same model but with log-Student's t:**
```
basedamage_hurricane_model_ec_p_tr_surg_f_alpha_lst
                                                  ^^^
```

**Same model but with Pareto:**
```
basedamage_hurricane_model_ec_p_tr_surg_f_alpha_pw
                                                 ^^
```

## Quick Reference

### Feature Abbreviations (existing)
```
ec      = exposure
p       = pressure
w       = wind
tr      = trend
surg_f  = surge_full
tid     = tides
sea     = seasonal
ib      = inverse_barometer
```

### Comparing Likelihoods

To see which likelihood performs best:

```python
# Same features, different likelihoods
model_specs = [
    {"exposure": True, "pressure": True, "wind": True, "likelihood": "lognormal"},
    {"exposure": True, "pressure": True, "wind": True, "likelihood": "log_studentt"},
    {"exposure": True, "pressure": True, "wind": True, "likelihood": "pareto"},
]

comparison_df, traces = compare_models(df_clean, model_specs)
print(comparison_df)
```

Output will show:
- `basedamage_hurricane_model_ec_p_w_ln`   (log-normal)
- `basedamage_hurricane_model_ec_p_w_lst`  (log-Student's t) ← probably best
- `basedamage_hurricane_model_ec_p_w_pw`   (Pareto)

## Usage Examples

### Default (Log-Normal)
```python
model_spec = {
    "exposure": True,
    "pressure": True,
}
# Results in: basedamage_hurricane_model_ec_p_ln
```

### Explicit Log-Normal
```python
model_spec = {
    "exposure": True,
    "pressure": True,
    "likelihood": "lognormal",
}
# Results in: basedamage_hurricane_model_ec_p_ln
```

### Heavy-Tailed (Log-Student's t)
```python
model_spec = {
    "exposure": True,
    "pressure": True,
    "likelihood": "log_studentt",
}
# Results in: basedamage_hurricane_model_ec_p_lst
```

### Power-Law (Pareto)
```python
model_spec = {
    "exposure": True,
    "pressure": True,
    "likelihood": "pareto",
}
# Results in: basedamage_hurricane_model_ec_p_pw
```

## Why This Matters

1. **Clarity**: Model name immediately shows which likelihood was used
2. **Comparison**: Easy to spot related models with different likelihoods
3. **Reproducibility**: Clear which version produced which results
4. **Thesis**: Neat argument: "Model X_lst beats X_ln by 8.3 elpd points"

## Reading Model Comparisons from CSV

When you see in `model_comparison_loo.csv`:

```
ec_p_tr_surg_f_alpha_lst,0,-4317.24,...   ← Best (log-Student's t)
ec_p_tr_surg_f_alpha_ln,1,-4325.54,...    ← Second (log-normal)
ec_p_tr_surg_f_alpha_pw,2,-4330.12,...    ← Third (Pareto)
```

You can instantly decode:
- All three use: exposure (ec), pressure (p), trend (tr), surge_full (surg_f), alpha
- lst won over ln and pw
- Difference: 8.3 points (ln vs lst), 12.9 points (pw vs lst)

## Thesis Statement Examples

"We tested three likelihoods with specification {exposure, pressure, wind}:
- `ec_p_w_ln` (log-normal): elpd_loo = -5200
- `ec_p_w_lst` (log-Student's t): elpd_loo = -5192 ✓ **8 points better**
- `ec_p_w_pw` (Pareto): elpd_loo = -5198

The log-Student's t distribution (LST) emerged as the best fit, indicating heavy-tailed but finite-variance damage distributions."

## File Organization

Model outputs are organized by model name:

```
Speciale/Code/Simulations/Plots/
├── basedamage_hurricane_model_ec_p_ln/
│   ├── ... (plots and outputs for lognormal)
├── basedamage_hurricane_model_ec_p_lst/
│   ├── ... (plots and outputs for log-Student's t)
├── basedamage_hurricane_model_ec_p_pw/
│   ├── ... (plots and outputs for Pareto)
```

This makes it easy to:
1. Find outputs for specific models
2. Compare likelihood effects
3. Track which version produced what results
