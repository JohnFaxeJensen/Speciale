import pymc as pm
import numpy as np
import arviz as az
from matplotlib import pyplot as plt
import os
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from wind_eq_comparison import equation_11_wind

import sys
sys.path.append(r"./Speciale/Code")
from merge_temp import merge_temp_data,merge_temp_data_monthly
from ULLN import ulln_logp, ulln_random

#Difference between this and the week 2 is that i try to include new model terms
#such as inverse barometer effect and using a model to transform pressure to wind speed

def hurricane_physical_model(df,  model_spec=None, ATD=False, inflation=False, use_weinkle_atd=False, use_ulln=False):
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
    
    if model_spec is None:
        raise ValueError("model_spec must be provided as a dictionary specifying model terms.")
    
    path = r"./Speciale/Code/Week2/Plots"
    # Clean data
    df_clean = df[df['basedamage'] > 0].copy()
    df_clean = df_clean.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure', 'ND'])
    #df_clean = df_clean.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure', 'ND', 'multiplier']) #testing

    #try weinkle ATD if specified

    # Prepare variables
    basedamage = df_clean['basedamage'].values
    
    if ATD:
        observed = np.log(df_clean['ATD'].values)
    if inflation:
        df_clean = df_clean[df_clean['ND'] > 0]
        observed = np.log(df_clean['ND'].values)
    if use_weinkle_atd:
        df_clean = df_clean.dropna(subset=['ATD_weinkle'])
        df_clean['ATD'] = df_clean['ATD_weinkle']
        observed = np.log(df_clean['ATD'].values)
    if not ATD and not inflation and not use_weinkle_atd:
        observed = np.log(basedamage)


    population = df_clean['population'].values
    WPC = df_clean['WPC'].values
    wind_speed_raw = df_clean['lf_wind'].values
    pressure_raw = df_clean['lf_pressure'].values
    area =10000 #value set by Aslak in study
    tides = df_clean['Tide_Level'].values
    tides_m = tides / 100  #convert cm to meters
    timestamps = df_clean['lf_ISO_TIME'].values 
    years = pd.to_datetime(timestamps).year
    years = [int(year - int(years[0])) for year in years]
    # Seasonal (day-of-year) features
    doy = pd.to_datetime(timestamps).dayofyear.values.astype(float)
    theta = 2 * np.pi * (doy / 365.25)
    season_sin = np.sin(theta)
    season_cos = np.cos(theta)

    travel_speed = df_clean['travel_speed_after_landfall_m_s'].values
    travel_speed_before = df_clean['travel_speed_before_landfall_m_s'].values

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

    multipliers = df_clean['multiplier'].values


    # Create model name based on spec
    spec_parts = [k for k, v in model_spec.items() if v]
    model_name = "_".join(spec_parts)
    if ATD:
        filename = f"ATD_hurricane_model_{model_name}"
    if inflation:
        filename = f"inflation_hurricane_model_{model_name}"
    if use_weinkle_atd:
        filename = f"weinkle_ATD_hurricane_model_{model_name}"
    if not ATD and not inflation and not use_weinkle_atd:
        filename = f"hurricane_model_{model_name}"
    model_path = os.path.join(r"./Speciale/Code/Week4/Plots", filename)
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
    
    ibtracks_speed = df_clean['ibtracs_speed_at_landfall_m_s'].values

    with pm.Model() as model:


        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=0, sigma=4)
        sigma = pm.HalfNormal("sigma", sigma=5)
        
        # Build mu dynamically based on model_spec
        mu = alpha
        
        if model_spec.get("economic", False):
            economic_coef = pm.Normal("economic_coef", sigma=1)
            mu = mu + economic_coef*np.log(population*WPC/area)

        if model_spec.get("economic_exp", False):
            economic_coef_exp = pm.Normal("economic_coef_exp", sigma=3)
            mu = mu + economic_coef_exp*(population*WPC/area)
        if model_spec.get("economic_atd", False):
            economic_coef_atd = pm.Normal("economic_coef_atd", sigma=1)
            mu = mu + economic_coef_atd*np.log(population*WPC* multipliers)
        

        if model_spec.get("economic_split", False):
            economic_coef_pop = pm.Normal("economic_coef_pop", sigma=3)
            economic_coef_wpc = pm.Normal("economic_coef_wpc", sigma=3)
            mu = mu + economic_coef_pop*np.log(population/area) + economic_coef_wpc*np.log(WPC)
        if model_spec.get("pressure", False):
            pressure_coef = pm.Normal("pressure_coef", sigma=3)
            mu = mu + pressure_coef*delta_P_relative
        
        if model_spec.get("tides", False):
            tides_coef = pm.Normal("tides_coef", sigma=3)
            mu = mu + tides_m*tides_coef
        
        if model_spec.get("wind", False):
            wind_coef = pm.Normal("wind_coef", sigma=3)
            mu = mu + wind_coef*wind_speed_relative

        if model_spec.get("modelled_wind", False):
            modelled_wind_coef = pm.Normal("modelled_wind_coef", sigma=3)
            mu = mu + modelled_wind_coef*modelled_wind_relative
            wind_coef_residual = pm.Normal("wind_coef_residual", sigma=3)
            mu = mu + wind_coef_residual*residual_wind_relative
        if model_spec.get("residual_wind", False):
            wind_coef_residual = pm.Normal("wind_coef_residual", sigma=3)
            mu = mu + wind_coef_residual*residual_wind_relative
        if model_spec.get("wind_power_law", False):
            wind_power_law_coef = pm.Normal("wind_power_law_coef", mu=9, sigma=4)
            wind_power_law = np.log(wind_speed_relative)*(wind_power_law_coef)
            mu = mu + wind_power_law
        
        if model_spec.get("travel_speed", False):
            travel_speed_coef = pm.Normal("travel_speed_coef", sigma=3)
            mu = mu + travel_speed_coef*travel_speed

        if model_spec.get("trend", False):
            trend_coef = pm.Normal("trend_coef", sigma=0.5)
            mu = mu + trend_coef*np.array(years)
        if model_spec.get("pca", False):
            pc1_coef = pm.Normal("pc1_coef", sigma=3)
            pc2_coef = pm.Normal("pc2_coef", sigma=3)
            mu = mu + pc1_coef*pc1 + pc2_coef*pc2

        if model_spec.get("inverse_barometer", False):
            ib_coef = pm.Normal("ib_coef", sigma=3)
            mu = mu + ib_coef*ib_vals_relative  # Example term for inverse barometer effect
        if model_spec.get("economic_raw", False):
            mu = mu + np.log(population*WPC/area)

        if model_spec.get("water_level", False):
            water_level_coef = pm.Normal("water_level_coef", sigma=3)
            water_level = tides_m + ib_vals_raw
            mu = mu + water_level_coef*water_level
        if model_spec.get("temp_anomaly_global", False):
            temp_anomaly_global_coef = pm.Normal("temp_anomaly_global_coef", sigma=3)
            mu = mu + temp_anomaly_global_coef*temp_anomaly_global
        if model_spec.get("seasonal", False):
            season_sin_coef = pm.Normal("season_sin_coef", sigma=3)
            season_cos_coef = pm.Normal("season_cos_coef", sigma=3)
            mu = mu + season_sin_coef * season_sin
            mu = mu + season_cos_coef * season_cos 
        if model_spec.get("pressure_trend_interaction", False):
            pressure_trend_interaction_coef = pm.Normal("pressure_trend_interaction_coef", sigma=3)
            mu = mu + pressure_trend_interaction_coef * delta_P_relative * np.array(years)
        if model_spec.get("gc_hadisst", False):
            temp_coef = pm.Normal("coef_gc_hadisst", sigma=3)
            mu = mu + temp_coef * temp_hadisst_gc
        if model_spec.get("mdr_hadisst", False):
            temp_coef_mdr = pm.Normal("coef_mdr_hadisst", sigma=3)
            mu = mu + temp_coef_mdr * temp_hadisst_mdr
        if model_spec.get("gc_icoads", False):
            temp_coef = pm.Normal("coef_gc_icoads", sigma=3)
            mu = mu + temp_coef * temp_icoads_gc
        if model_spec.get("mdr_icoads", False):
            temp_coef_mdr = pm.Normal("coef_mdr_icoads", sigma=3)
            mu = mu + temp_coef_mdr * temp_icoads_mdr
        if model_spec.get("sea_level_rise", False):
            slope_sea_level = 0.0025  # meters per year
            sea_level_rise = slope_sea_level * np.array(years)
            sea_level_coef = pm.Normal("sea_level_coef", sigma=3)
            mu = mu + sea_level_coef * sea_level_rise
        if model_spec.get("ibtracks_speed", False):
            ibtracks_speed_coef = pm.Normal("ibtracks_speed_coef", sigma=3)
            mu = mu + ibtracks_speed_coef * (ibtracks_speed / 10)  # normalize to 10 m/s
        
        if model_spec.get("wind_vulnerability", False):
            v_threshold = pm.Normal("v_threshold", mu=50, sigma=10)
            v_half = pm.Normal("v_half", mu=120, sigma=15)
            v_coef = pm.Normal("v_coef", sigma=3)
            v_n = pm.math.maximum(wind_speed_raw-v_threshold,0)/(v_half-v_threshold)
            vulnerability = v_n**3/(1+v_n**3)
            mu = mu + v_coef*np.log(area*vulnerability + 1e-6)  # add small constant to avoid log(0)
        


        if use_ulln:
            upper_min = np.max(np.exp(observed)) 
            upper_min_prior = pm.Pareto("upper", alpha=2.0, m=upper_min)
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
    if model_spec.get("economic_atd", False):
        var_names.append("economic_coef_atd")
    if model_spec.get("ibtracks_speed", False):
        var_names.append("ibtracks_speed_coef")
    if model_spec.get("wind_vulnerability", False):
        var_names += ["v_threshold", "v_half", "v_coef"]



    

    

    

    
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
    combined = np.concatenate([observed, ppc_values])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(observed))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(observed, bins=bins, density=True, alpha=0.5, label="Observed")
    if use_ulln:
        plt.hist(np.log(ppc_values), bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    else:
        plt.hist(ppc_values, bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title(f"Posterior Predictive Histogram ({model_name})")
    plt.legend()
    plt.savefig(os.path.join(model_path, f"{filename}_ppc_histogram.png"))
    plt.close()
    
    return trace, model_name





def compare_models(df, model_specs, ATD=False, inflation=False, use_weinkle_atd=False, use_ulln=False):
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
        trace, model_name = hurricane_physical_model(df, model_spec=spec, ATD=ATD, inflation=inflation , use_weinkle_atd=use_weinkle_atd, use_ulln=use_ulln)
        traces[model_name] = trace

    
    # Compare models using WAIC
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (WAIC)")
    print("="*80)
    comparison_df = az.compare(traces)
    print(comparison_df)
    #az.plot_compare(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(r"./Speciale/Code/Week4/Plots/model_comparison_waic.csv")
    
    # Also try LOO
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (LOO)")
    print("="*80)
    try:
        comparison_loo = az.compare(traces, ic="loo")
        az.plot_compare(comparison_loo)
        plt.savefig(r"./Speciale/Code/Week4/Plots/model_comparison_loo.png")
        plt.show()

        print(comparison_loo)
        comparison_loo.to_csv(r"./Speciale/Code/Week4/Plots/model_comparison_loo.csv")
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

    df = pd.read_csv('./Speciale/Hurricane_data/Aslak_data_with_tide_and_travelspeed.csv')
    #first merge global temp data
    df_temp_data = pd.read_csv('./Speciale/temp_data/data_global_ocean_temp.csv', skiprows=3)
    df = merge_temp_data(df, df_temp_data, 'Year')
    #then merge regional hadisst temp data
    df_temp_data = pd.read_csv('./Speciale/temp_data/mean_sst_regions_HadISST.csv')
    df = merge_temp_data_monthly(df, df_temp_data, ['Year', 'Month'])
    #then merge regional icoads temp data
    df_temp_data = pd.read_csv('./Speciale/temp_data/mean_sst_regions_Icoads.csv')
    df = merge_temp_data_monthly(df, df_temp_data, ['Year', 'Month'])
    #try the weinkle basedamage data. reformat to merge with main df
    df_weinkle_data = pd.read_excel('./Speciale/Hurricane_data/Aslak_data.xls', sheet_name='ATD of Weinkle')
    atd_weinkle = df_weinkle_data[['ATCF_ID', 'ATD', 'basedamage', 'ND']]
    atd_weinkle['multiplier'] = atd_weinkle['ND'] / atd_weinkle['basedamage']
    rename_dict = {'ATD': 'ATD_weinkle', 'basedamage': 'basedamage_weinkle', 'ND': 'ND_weinkle'}
    atd_weinkle.rename(rename_dict, axis=1, inplace=True)

    df = pd.merge(df, atd_weinkle, how='left', left_on='ATCF_ID', right_on='ATCF_ID')
    df_clean = df.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    df_clean = df_clean[df_clean['basedamage'] > 0]
    df_clean = df_clean[df_clean['ND'] > 0]
    BD = np.array(df_clean['basedamage'].values)
    ND = np.array(df_clean['ND'].values)
    
    

    
    #Here is the wind models comparisons using models from the paper:
    #https://journals.ametsoc.org/view/journals/mwre/136/9/2008mwr2395.1.pdf
    # Your data
    
    #test_pressure_to_wind_models()
    #quit()
    # pressure = df_clean['lf_pressure'].values
    # wind = df_clean['lf_wind'].values
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.scatter(np.log(ND), wind, alpha=0.6)
    # plt.xlabel("ln(Base Damage)")
    # plt.ylabel("Wind Speed (knots)")
    # plt.show()
    # plt.scatter(np.log(ND), pressure, alpha=0.6)
    # plt.xlabel("ln(Base Damage)")
    # plt.ylabel("Pressure (mb)")
    # plt.show()
    # quit()
    # Define model specifications to compare
    model_specs = [
        {"economic": True,  "pressure": True},
        {"economic": True, "pressure": True, "wind_power_law": True},
        {"economic": True,  "wind_power_law": True},
        #{"economic": True, "modelled_wind": True},
        # {"economic_atd": True, "trend": True, "modelled_wind": True, "travel_speed": True,  "mdr_icoads": True},
        # {"economic_atd": True, "trend": True, "modelled_wind": True, "seasonal":True},
        # {"economic_atd": True, "modelled_wind": True, "trend": True},


    ]

    # model_specs = [            # Wind only
    # {"pressure": True, "trend": True, "economic": True},          # Pressure only
    # ]


    # # # Then run period comparison on EACH
    # for spec in model_specs:
    #     traces = compare_model_by_period(df, spec, cutoff_year=1950, ATD=False, inflation=False, use_weinkle_atd=False, use_ulln=False)
    # Run comparison
    df_clean = df_clean[df_clean['Year'] >= 2003]
    df_clean = df_clean.dropna(subset=['rmw_at_landfall_m'])

    # Convert to float first, then drop empty strings
    df_clean['rmw_at_landfall_m'] = pd.to_numeric(df_clean['rmw_at_landfall_m'], errors='coerce')
    df_clean = df_clean.dropna(subset=['rmw_at_landfall_m'])

    # Now plot
    plt.scatter( df_clean['rmw_at_landfall_m'], np.log(df_clean['basedamage']))
    plt.show()
    quit()
    comparison_df, traces = compare_models(df, model_specs, ATD=False, inflation=True, use_weinkle_atd=False, use_ulln=False)
        # Compare both methods


# Now merge with your main dataframe:
# df = pd.merge(df, df_surge, left_on='ATCF_ID', right_on='stname', how='left')
    # df_post1960 = df_clean[df_clean['Year'] >= 1960]
    # corr_pre = df_pre1960[['lf_wind', 'lf_pressure']].corr().iloc[0,1]
    # corr_post = df_post1960[['lf_wind', 'lf_pressure']].corr().iloc[0,1]
    # print(f"Correlation (Wind vs Pressure) pre-1960: {corr_pre:.3f}")
    # print(f"Correlation (Wind vs Pressure) post-1960: {corr_post:.3f}")
    # #check correlation between wind and pressure pre and post 1960 by wind equation how well it models wind from pressure
    # modelled_wind_pre = equation_11_wind(df_pre1960['lf_pressure'].values, df_pre1960['lf_lat'].values, df_pre1960['travel_speed_after_landfall_m_s'].values)
    # modelled_wind_post = equation_11_wind(df_post1960['lf_pressure'].values, df_post1960['lf_lat'].values, df_post1960['travel_speed_after_landfall_m_s'].values)
    # print(f"R^2 (Modelled Wind vs Observed Wind) pre-1960: {np.corrcoef(modelled_wind_pre, df_pre1960['lf_wind'].values)[0,1]**2:.3f}")
    # print(f"R^2 (Modelled Wind vs Observed Wind) post-1960: {np.corrcoef(modelled_wind_post, df_post1960['lf_wind'].values)[0,1]**2:.3f}")
