import pymc as pm
import numpy as np
import arviz as az
from matplotlib import pyplot as plt
import os
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sys
sys.path.append(r"./Speciale/Code")
from wind_eq_comparison_final import equation_11_wind
from ULLN import ulln_logp, ulln_random
from Data_preprocessing.generate_csv import generate_csv_data






def hurricane_physical_model(df,  model_spec=None, use_ulln=False, observed_variable='basedamage'):
    """
    Build and fit a hurricane damage model with configurable terms.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    model_spec : dict, optional
        Specify which terms to include. If None, uses full model.
        Example: {"economic": True, "pressure": True, "tides": False, "wind": True, "travel_speed": False}
    
    Returns:
    --------
    trace : arviz InferenceData
        MCMC trace from sampling
    model_name : str
        Descriptive name of the model specification
    """
    valid_observed_vars = ['basedamage', 'ATD', 'ND']
    if observed_variable not in valid_observed_vars:
        raise ValueError(f"observed_variable must be one of {valid_observed_vars}")
    
    if model_spec is None:
        raise ValueError("model_spec must be provided as a dictionary specifying model terms.")
    df_clean = df.dropna(subset=[observed_variable, 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    df_clean = df_clean[df_clean[observed_variable] > 0].copy()
    # Clean data
    observed = df_clean[observed_variable].values
    # transform to log-space
    observed = np.log(observed)



    population = df_clean['population'].values
    WPC = df_clean['WPC'].values
    wind_speed_raw = df_clean['lf_wind'].values
    pressure_raw = df_clean['lf_pressure'].values
    area =10000 #value set by Aslak in study
    tides_m = df_clean['Tide_Level_m'].values
    timestamps = df_clean['lf_ISO_TIME'].values 
    years = pd.to_datetime(timestamps).year
    years = [int(year - int(years[0])) for year in years]
    # Seasonal (day-of-year) features
    doy = pd.to_datetime(timestamps).dayofyear.values.astype(float)
    theta = 2 * np.pi * (doy / 365.25)
    season_sin = np.sin(theta)
    season_cos = np.cos(theta)

    travel_speed = df_clean['lf_speed_after_ms'].values
    travel_speed_before = df_clean['lf_speed_before_ms'].values

    # load and detrend anomaly 
    temp_anomaly_global = df_clean['Anomaly_global'].values
    temp_anomaly_global = temp_anomaly_global - (np.array(years)-years[0])*0.008 #rough detrend

    modelled_wind_raw = equation_11_wind(pressure_raw, df_clean['lf_lat'].values, travel_speed)
    # Orthogonalize residuals: wind_speed ~ modelled_wind, use residual of this regression
    lr = LinearRegression().fit(modelled_wind_raw.reshape(-1, 1), wind_speed_raw)
    wind_pred_raw = lr.predict(modelled_wind_raw.reshape(-1, 1))
    residual_wind_raw = wind_speed_raw - wind_pred_raw
    temp_hadisst_gc = df_clean['mean_temp_hadisst_gc'].values
    temp_hadisst_mdr = df_clean['mean_temp_hadisst_mdr'].values
    temp_icoads_gc = df_clean['mean_temp_icoads_gc'].values
    temp_icoads_mdr = df_clean['mean_temp_icoads_mdr'].values
    baseline_decade = (np.array(years) >= 70) & (np.array(years) < 80) 
    temp_hadisst_gc_anom = temp_hadisst_gc - temp_hadisst_gc[baseline_decade].mean()
    temp_hadisst_mdr_anom = temp_hadisst_mdr - temp_hadisst_mdr[baseline_decade].mean()
    temp_icoads_gc_anom = temp_icoads_gc - temp_icoads_gc[baseline_decade].mean()
    temp_icoads_mdr_anom = temp_icoads_mdr - temp_icoads_mdr[baseline_decade].mean()
    risk_scores = df_clean['Hurricane_Risk_Score'].values



    # Create model name based on spec
    spec_parts = [k for k, v in model_spec.items() if v]
    model_name = "_".join(spec_parts)
    filename = f"{observed_variable}_hurricane_model_{model_name}"
    if use_ulln:
        filename += "_ulln"
    model_path = os.path.join(r"./Speciale/Code/Simulations/Plots", filename)
    os.makedirs(model_path, exist_ok=True)

    category_1_wind_baseline = 95*0.868976242 #convert mph to knots
    category_1_pressure_baseline = 980 #mb
    wind_speed_relative = wind_speed_raw / category_1_wind_baseline
    pressure_relative = pressure_raw / category_1_pressure_baseline
    travel_speed = travel_speed / 10 #normalize to 10 m/s
    travel_speed_before = travel_speed_before / 10 #normalize to 10 m/s
    modelled_wind_relative = modelled_wind_raw / category_1_wind_baseline
    residual_wind_relative = residual_wind_raw / category_1_wind_baseline

    # Standardize pressure and wind
    scaler = StandardScaler()
    pressure_wind_scaled = scaler.fit_transform(np.column_stack([pressure_relative, wind_speed_relative]))

    # Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(pressure_wind_scaled)

    # Use PCA components in your model
    pc1, pc2 = pca_components[:, 0], pca_components[:, 1]
    
    #use delta_P as pressure proxy
    delta_P = (1013.25-pressure_raw)* 100  #convert to Pa
    delta_P_baseline = (1013.25 - category_1_pressure_baseline)*100
    delta_P_relative = delta_P / delta_P_baseline  #Pa
    #Inverse barometer effect calculation
    density_water = 1000  # kg/m³
    g = 9.81  # m/s²
    ib_vals_raw = delta_P / (density_water * g)

    category_1_ib_baseline = (1013.25 - category_1_pressure_baseline) * 100 / (density_water * g)
    ib_vals_relative = ib_vals_raw / category_1_ib_baseline
    
    ibtracks_speed = df_clean['STORM_SPEED_ms'].values
    storm_tides_m = df_clean['Storm_Tide_m'].values

    r34_mean = df_clean['R34_MEAN_RADIUS'].values
    r50_mean = df_clean['R50_MEAN_RADIUS'].values
    r64_mean = df_clean['R64_MEAN_RADIUS'].values

    economic_var = np.log(population*WPC/area)
    lr = LinearRegression().fit(economic_var.reshape(-1, 1), np.array(years))
    years_residual = np.array(years) - lr.predict(economic_var.reshape(-1, 1))
    with pm.Model() as model:


        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=0, sigma=15)
        sigma = pm.HalfNormal("sigma", sigma=5)
        
        # Build mu dynamically based on model_spec
        mu = alpha
        
        if model_spec.get("economic", False):
            economic_coef = pm.Normal("economic_coef", sigma=5)
            mu = mu + economic_coef*np.log(population*WPC/area)

        if model_spec.get("economic_exp", False):
            economic_coef_exp = pm.Normal("economic_coef_exp", sigma=5)
            mu = mu + economic_coef_exp*(population*WPC/area)


        if model_spec.get("economic_split", False):
            economic_coef_pop = pm.Normal("economic_coef_pop", sigma=5)
            economic_coef_wpc = pm.Normal("economic_coef_wpc", sigma=5)
            mu = mu + economic_coef_pop*np.log(population/area) + economic_coef_wpc*np.log(WPC)
        if model_spec.get("pressure", False):
            # Linear in log-space: log(damage) ~ coef * pressure
            pressure_coef = pm.Normal("pressure_coef", sigma=5)
            mu = mu + pressure_coef*delta_P_relative
        
        if model_spec.get("pressure_threshold", False):
            # Threshold response in log-space: only log(pressure) above threshold
            pressure_thresh = pm.Normal("pressure_threshold_val", mu=-1.0, sigma=0.5)
            pressure_thresh_coef = pm.Normal("pressure_thresh_coef", sigma=5)
            pressure_log = pm.math.log(delta_P_relative + 0.1)
            pressure_excess = pm.math.maximum(pressure_log - pressure_thresh, 0)
            mu = mu + pressure_thresh_coef * pressure_excess
        if model_spec.get("pressure_threshold_2", False):
            # Power law above threshold: damage ~ (delta_P_relative - threshold)^alpha
            # mu = alpha + coef * max(delta_P_relative - threshold, eps)^power
            pressure_threshold = pm.Normal("pressure_threshold", mu=1.0, sigma=0.3)
            pressure_power = pm.Normal("pressure_power", mu=2.0, sigma=0.5)
            pressure_thresh_coef = pm.Normal("pressure_thresh_coef", sigma=5)
            pressure_excess = pm.math.maximum(delta_P_relative - pressure_threshold, 1e-6)
            power_term = pressure_excess**pressure_power
            mu = mu + pressure_thresh_coef * power_term

        
        if model_spec.get("tides", False):
            tides_coef = pm.Normal("tides_coef", sigma=5)
            mu = mu + tides_m*tides_coef
        
        if model_spec.get("wind", False):
            wind_coef = pm.Normal("wind_coef", sigma=5)
            mu = mu + wind_coef*wind_speed_relative

        if model_spec.get("modelled_wind", False):
            modelled_wind_coef = pm.Normal("modelled_wind_coef", sigma=5)
            mu = mu + modelled_wind_coef*modelled_wind_relative
            wind_coef_residual = pm.Normal("wind_coef_residual", sigma=5)
            mu = mu + wind_coef_residual*residual_wind_relative
        if model_spec.get("residual_wind", False):
            wind_coef_residual = pm.Normal("wind_coef_residual", sigma=5)
            mu = mu + wind_coef_residual*residual_wind_relative
        if model_spec.get("wind_power_law", False):
            wind_power_law_coef = pm.Normal("wind_power_law_coef", mu=9, sigma=4)
            wind_power_law = np.log(wind_speed_relative)*(wind_power_law_coef)
            mu = mu + wind_power_law
        
        if model_spec.get("travel_speed", False):
            travel_speed_coef = pm.Normal("travel_speed_coef", sigma=5)
            mu = mu + travel_speed_coef*travel_speed

        if model_spec.get("trend", False):
            trend_coef = pm.Normal("trend_coef", mu=0, sigma=0.1)
            mu = mu + trend_coef*np.array(years)
        if model_spec.get("trend_residual", False):
            trend_residual_coef = pm.Normal("trend_residual_coef", mu=0, sigma=0.1)
            mu = mu + trend_residual_coef*years_residual
        if model_spec.get("pca", False):
            pc1_coef = pm.Normal("pc1_coef", sigma=5)
            pc2_coef = pm.Normal("pc2_coef", sigma=5)
            mu = mu + pc1_coef*pc1 + pc2_coef*pc2

        if model_spec.get("inverse_barometer", False):
            ib_coef = pm.Normal("ib_coef", sigma=5)
            mu = mu + ib_coef*ib_vals_relative  # Example term for inverse barometer effect
        if model_spec.get("economic_raw", False):
            mu = mu + np.log(population*WPC/area)

        if model_spec.get("water_level", False):
            water_level_coef = pm.Normal("water_level_coef", sigma=5)
            water_level = tides_m + ib_vals_raw
            mu = mu + water_level_coef*water_level
        if model_spec.get("temp_anomaly_global", False):
            temp_anomaly_global_coef = pm.Normal("temp_anomaly_global_coef", sigma=5)
            mu = mu + temp_anomaly_global_coef*temp_anomaly_global
        if model_spec.get("seasonal", False):
            season_sin_coef = pm.Normal("season_sin_coef", sigma=5)
            season_cos_coef = pm.Normal("season_cos_coef", sigma=5)
            mu = mu + season_sin_coef * season_sin
            mu = mu + season_cos_coef * season_cos 
        if model_spec.get("pressure_trend_interaction", False):
            pressure_trend_interaction_coef = pm.HalfNormal("pressure_trend_interaction_coef", sigma=5)
            mu = mu + pressure_trend_interaction_coef * delta_P_relative * np.array(years)
        if model_spec.get("gc_hadisst", False):
            temp_coef = pm.Normal("coef_gc_hadisst", sigma=5)
            mu = mu + temp_coef * temp_hadisst_gc
        if model_spec.get("mdr_hadisst", False):
            temp_coef_mdr = pm.Normal("coef_mdr_hadisst", sigma=5)
            mu = mu + temp_coef_mdr * temp_hadisst_mdr
        if model_spec.get("gc_icoads", False):
            temp_coef = pm.Normal("coef_gc_icoads", sigma=5)
            mu = mu + temp_coef * temp_icoads_gc
        if model_spec.get("mdr_icoads", False):
            temp_coef_mdr = pm.Normal("coef_mdr_icoads", sigma=5)
            mu = mu + temp_coef_mdr * temp_icoads_mdr
        if model_spec.get("sea_level_rise", False):
            slope_sea_level = 0.0025  # meters per year
            sea_level_rise = slope_sea_level * np.array(years)
            sea_level_coef = pm.Normal("sea_level_coef", sigma=5)
            mu = mu + sea_level_coef * sea_level_rise
        if model_spec.get("ibtracks_speed", False):
            ibtracks_speed_coef = pm.Normal("ibtracks_speed_coef", sigma=5)
            mu = mu + ibtracks_speed_coef * (ibtracks_speed / 10)  # normalize to 10 m/s
        
        if model_spec.get("wind_vulnerability", False):
            v_threshold = pm.Normal("v_threshold", mu=50, sigma=10)
            v_half = pm.Normal("v_half", mu=120, sigma=15)
            v_coef = pm.Normal("v_coef", sigma=5)
            v_n = pm.math.maximum(wind_speed_raw-v_threshold,0)/(v_half-v_threshold)
            vulnerability = v_n**3/(1+v_n**3)
            mu = mu + v_coef*np.log(area*vulnerability + 1e-6)  # add small constant to avoid log(0)
        if model_spec.get('r34_mean', False):
            r34_coef = pm.Normal("r34_coef", sigma=5)
            mu = mu + r34_coef * r34_mean
        if model_spec.get('r50_mean', False):
            r50_coef = pm.Normal("r50_coef", sigma=5)
            mu = mu + r50_coef * r50_mean
        if model_spec.get('r64_mean', False):
            r64_coef = pm.Normal("r64_coef", sigma=5)
            mu = mu + r64_coef * r64_mean
        if model_spec.get("vulnerability", False):
            # Calculate damage normalized by exposed value in each state
            # damage_to_exposure = basedamage / (population * WPC)
            # This removes economic growth confounding
            basedamage = df_clean['basedamage'].values
            damage_to_exposure = basedamage / (population * WPC / area)
            
            # Add to dataframe so groupby can use it
            df_clean_temp = df_clean.copy()
            df_clean_temp['damage_to_exposure'] = damage_to_exposure
            
            # Get median damage-to-exposure by state
            state_damage_exposure = df_clean_temp.groupby('lf_state')['damage_to_exposure'].mean().to_dict()
            print(state_damage_exposure)
            states = df_clean['lf_state'].values
            state_vulnerability = np.array([state_damage_exposure.get(s, 1.0) for s in states])
            state_vulnerability = (state_vulnerability - state_vulnerability.mean()) / state_vulnerability.std()
            
            state_vuln_coef = pm.Normal("state_vuln_coef", sigma=5)
            mu = mu + state_vuln_coef * state_vulnerability
        if model_spec.get("risk_score", False):
            risk_score_coef = pm.Normal("risk_score_coef", sigma=5)
            mu = mu + risk_score_coef * risk_scores
        if model_spec.get("storm_tide", False):
            storm_tide_coef = pm.HalfNormal("storm_tide_coef", sigma=5)
            mu = mu + storm_tide_coef * storm_tides_m
        if model_spec.get("storm_tide_threshold", False):
            storm_tide_thresh = pm.Normal("storm_tide_thresh", mu=0.5, sigma=0.2)
            storm_tide_thresh_coef = pm.HalfNormal("storm_tide_thresh_coef", sigma=5)
            storm_tide_excess = pm.math.maximum(storm_tides_m - storm_tide_thresh, 0)
            mu = mu + storm_tide_thresh_coef * storm_tide_excess




        if use_ulln:
            upper_min = np.max(np.exp(observed)) 
            upper_min_prior = pm.Pareto("upper", alpha=5.0, m=upper_min)
            obs = pm.CustomDist("obs",
                                 mu, sigma, upper_min_prior,
                                 logp=ulln_logp,
                                 random=ulln_random,
                                 observed=np.exp(observed))
        else:  
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
        
        trace = pm.sample(draws=2500, tune=1000, target_accept=0.95, idata_kwargs={'log_likelihood':True})
        summary = az.summary(trace, hdi_prob=0.95)
        summary.to_csv(os.path.join(model_path, f"{filename}_summary.csv"))
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(summary)

    # Plot traces (only for included variables)
    var_names = ["alpha", "sigma"]
    if model_spec.get("economic", False):
        var_names.append("economic_coef")
    if model_spec.get("pressure", False):
        var_names.append("pressure_coef")
    if model_spec.get("pressure_power_law", False):
        var_names.append("pressure_power_coef")
    if model_spec.get("pressure_sqrt", False):
        var_names.append("pressure_sqrt_coef")
    if model_spec.get("pressure_quadratic", False):
        var_names.append("pressure_quad_coef")
    if model_spec.get("pressure_log", False):
        var_names.append("pressure_log_coef")
    if model_spec.get("pressure_wind_interaction", False):
        var_names.append("pressure_wind_int_coef")
    if model_spec.get("pressure_wind_power", False):
        var_names.append("pressure_wind_power_coef")
    if model_spec.get("pressure_threshold", False):
        var_names.append("pressure_threshold_val")
        var_names.append("pressure_thresh_coef")
    if model_spec.get("pressure_saturation", False):
        var_names.append("pressure_sat_coef")
        var_names.append("pressure_sat_point")
    if model_spec.get("tides", False):
        var_names.append("tides_coef")
    if model_spec.get("wind", False):
        var_names.append("wind_coef")
    if model_spec.get("wind_power_law", False):
        var_names.append("wind_power_law_coef")
    if model_spec.get("modelled_wind", False):
        var_names.append("modelled_wind_coef")
        var_names.append("wind_coef_residual")
    if model_spec.get("travel_speed", False):
        var_names.append("travel_speed_coef")
    if model_spec.get("trend", False):
        var_names.append("trend_coef")
    if model_spec.get("pca", False):
        var_names.append("pc1_coef")
        var_names.append("pc2_coef")
    if model_spec.get("inverse_barometer", False):
        var_names.append("ib_coef")
    if model_spec.get("water_level", False):
        var_names.append("water_level_coef")
    if model_spec.get("residual_wind", False):
        var_names.append("wind_coef_residual")
    if model_spec.get("temp_anomaly_global", False):
        var_names.append("temp_anomaly_global_coef")
    if model_spec.get("seasonal", False):
        var_names += ["season_sin_coef", "season_cos_coef"]
    if model_spec.get("pressure_trend_interaction", False):
        var_names.append("pressure_trend_interaction_coef")
    if model_spec.get("gc_hadisst", False):
        var_names.append("coef_gc_hadisst")
    if model_spec.get("mdr_hadisst", False):
        var_names.append("coef_mdr_hadisst")
    if model_spec.get("gc_icoads", False):
        var_names.append("coef_gc_icoads")
    if model_spec.get("mdr_icoads", False):
        var_names.append("coef_mdr_icoads")
    if model_spec.get("economic_split", False):
        var_names += ["economic_coef_pop", "economic_coef_wpc"]
    if model_spec.get("sea_level_rise", False):
        var_names.append("sea_level_coef")
    if model_spec.get("ibtracks_speed", False):
        var_names.append("ibtracks_speed_coef")
    if model_spec.get("wind_vulnerability", False):
        var_names += ["v_threshold", "v_half", "v_coef"]
    if use_ulln:
        var_names.append("upper")
    if model_spec.get('r34_mean', False):
        var_names.append("r34_coef")
    if model_spec.get('r50_mean', False):
        var_names.append("r50_coef")
    if model_spec.get('r64_mean', False):
        var_names.append("r64_coef")
    if model_spec.get("vulnerability", False):
        var_names.append("state_vuln_coef")
    if model_spec.get("trend_residual", False):
        var_names.append("trend_residual_coef")
    if model_spec.get("pressure_threshold_2", False):
        var_names += ["pressure_threshold", "pressure_power", "pressure_thresh_coef"]
    if model_spec.get("risk_score", False):
        var_names.append("risk_score_coef")
    if model_spec.get("storm_tide", False):
        var_names.append("storm_tide_coef")




    

    

    

    
    az.plot_trace(trace, var_names=var_names, figsize=(12, 12))
    plt.savefig(os.path.join(model_path, f"{filename}_trace.png"))
    plt.close()
    
    az.plot_pair(trace, var_names=var_names, kind="kde", marginals=True, divergences=True)
    plt.savefig(os.path.join(model_path, f"{filename}_pair.png"))
    plt.close()
    
    # Posterior predictive check
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    
    ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    # Histogram comparison
    if use_ulln:
        # ppc_values are in original scale, observed is in log scale
        ppc_values_log = np.log(ppc_values)
        combined = np.concatenate([observed, ppc_values_log])
        bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(observed))), 25))
        plt.figure(figsize=(10,6))
        plt.hist(observed, bins=bins, density=True, alpha=0.5, label="Observed")
        plt.hist(ppc_values_log, bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    else:
        # Both are already in log scale
        combined = np.concatenate([observed, ppc_values])
        bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(observed))), 25))
        plt.figure(figsize=(10,6))
        plt.hist(observed, bins=bins, density=True, alpha=0.5, label="Observed")
        plt.hist(ppc_values, bins=bins, density=True, alpha=0.5, label="Posterior predictive")

    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title(f"Posterior Predictive Histogram ({model_name})")
    plt.legend()
    plt.savefig(os.path.join(model_path, f"{filename}_ppc_histogram.png"))
    plt.close()
    
    return trace, model_name





def compare_models(df, model_specs, use_ulln=False, observed_variable='basedamage'):
    """
    Compare multiple model specifications using WAIC and LOO.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    model_specs : list of dicts
        List of model specification dicts, each with keys: {"economic": bool, "pressure": bool, ...}
    
    Returns:
    --------
    comparison_df : DataFrame
        Comparison results ranked by model performance
    traces : dict
        Dictionary mapping model names to their traces
    """
    traces = {}
    
    print("\n" + "="*80)
    print("FITTING MULTIPLE MODELS FOR COMPARISON")
    print("="*80 + "\n")
    
    for i, spec in enumerate(model_specs, 1):
        print(f"\n[{i}/{len(model_specs)}] Fitting model with spec: {spec}")
        trace, model_name = hurricane_physical_model(df, model_spec=spec, use_ulln=use_ulln, observed_variable=observed_variable)
        traces[model_name] = trace

    
    # Compare models using WAIC
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (WAIC)")
    print("="*80)
    comparison_df = az.compare(traces)
    print(comparison_df)
    #az.plot_compare(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(r"./Speciale/Code/Simulations/model_comparison_waic.csv")
    
    # Also try LOO
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (LOO)")
    print("="*80)
    try:
        comparison_loo = az.compare(traces, ic="loo")
        az.plot_compare(comparison_loo)
        plt.savefig(r"./Speciale/Code/Simulations/model_comparison_loo.png")
        plt.show()

        print(comparison_loo)
        comparison_loo.to_csv(r"./Speciale/Code/Simulations/model_comparison_loo.csv")
    except Exception as e:
        print(f"LOO comparison failed (may need pareto_k adjustment): {e}")
    
    return comparison_df, traces


def compare_model_by_period(df, model_spec, cutoff_year, ATD=False, inflation=False, use_weinkle_atd=False, use_ulln=False):
    """
    Fit same model to before/after periods and compare sigma, coefficients, fit quality.
    """
    df_before = df[df['Year'] < cutoff_year].copy()
    df_after = df[df['Year'] >= cutoff_year].copy()
    
    print(f"\n{'='*80}")
    print(f"BEFORE {cutoff_year}: {len(df_before)} observations")
    print(f"AFTER {cutoff_year}: {len(df_after)} observations")
    print(f"{'='*80}\n")
    
    traces = {}
    
    for period_name, df_period in [("before", df_before), ("after", df_after)]:
        print(f"\n--- Fitting model for {period_name} {cutoff_year} ---")
        trace, model_name = hurricane_physical_model(df_period, model_spec=model_spec, ATD=ATD, inflation=inflation, use_weinkle_atd=use_weinkle_atd, use_ulln=use_ulln)
        traces[period_name] = trace

    
    return traces

    



if __name__ == "__main__":
    #Example usage
    df = generate_csv_data()
    print(df.shape)
    df_clean = df.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure', 'Storm_Tide_m'])
    print(df_clean.shape)
    df_clean = df_clean[df_clean['basedamage'] > 0]
    print(df_clean.shape)
    df_clean = df_clean[df_clean['ND'] > 0]
    print(df_clean.shape)



    model_specs = [
        {"economic": True, "pressure": True, "storm_tide": True},
        {"economic": True, "pressure": True},
   
  

    ]
    #hurricane_physical_model(df_clean, model_spec={"economic": True,  "pressure": True, "storm_tide": True},use_ulln=False, observed_variable='ATD')
    comparison_df, traces = compare_models(df_clean, model_specs, use_ulln=True, observed_variable='basedamage')
        # Compare both methods

