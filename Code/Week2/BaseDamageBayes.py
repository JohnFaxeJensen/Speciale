import pymc as pm
import numpy as np
import scipy.stats as stats
import arviz as az
from matplotlib import pyplot as plt
import os
import pandas as pd
from pytensor.printing import Print

import statsmodels.api as sm




def hurricane_physical_model(df, use_ND=False, model_spec=None):
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
        model_spec = {"economic": True, "pressure": True, "tides": True, "wind": True, "travel_speed": True}
    
    path = r"./Speciale/Code/Week2/Plots"

    # Clean data
    df_clean = df[df['basedamage'] > 0].copy()
    df_clean = df_clean.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    # Prepare variables
    observed = np.log(df_clean['basedamage'])
    if use_ND:
        df_clean = df_clean[df_clean['ND'] > 0].copy()
        observed = np.log(df_clean['ND'])

    population = df_clean['population'].values
    WPC = df_clean['WPC'].values
    wind_speed = df_clean['lf_wind'].values
    pressure = df_clean['lf_pressure'].values
    area =10000 #value set by Aslak in study
    tides = df_clean['Tide_Level'].values
    timestamps = df_clean['lf_ISO_TIME'].values 
    years = pd.to_datetime(timestamps).year
    years = [int(year - int(years[0])) for year in years]
    travel_speed = df_clean['travel_speed_after_landfall_m_s'].values

    # Create model name based on spec
    spec_parts = [k for k, v in model_spec.items() if v]
    model_name = "full" if len(spec_parts) == len(model_spec) else "_".join(spec_parts)
    
    filename = f"hurricane_model_{model_name}"
    model_path = os.path.join(r"./Speciale/Code/Week2/Plots", filename)
    os.makedirs(model_path, exist_ok=True)

    category_1_wind_baseline = 95*0.868976242 #convert mph to knots
    category_1_pressure_baseline = 980 #mb
    wind_speed = wind_speed / category_1_wind_baseline
    pressure = pressure / category_1_pressure_baseline
    travel_speed = travel_speed / 10 #normalize to 10 m/s

    with pm.Model() as model:
        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=17, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        
        # Build mu dynamically based on model_spec
        mu = alpha
        
        if model_spec.get("economic", False):
            economic_coef = pm.Normal("economic_coef", sigma=2)
            mu = mu + economic_coef*np.log(population*WPC/area)
        
        if model_spec.get("pressure", False):
            pressure_coef = pm.Normal("pressure_coef", sigma=3)
            mu = mu + pressure*pressure_coef
        
        if model_spec.get("tides", False):
            tides_coef = pm.Normal("tides_coef", sigma=2)
            mu = mu + tides*tides_coef
        
        if model_spec.get("wind", False):
            wind_speed_coef = pm.Normal("wind_speed_coef", sigma=2)
            mu = mu + wind_speed*wind_speed_coef
        
        if model_spec.get("travel_speed", False):
            travel_speed_coef = pm.Normal("travel_speed_coef", sigma=2)
            mu = mu + travel_speed_coef*travel_speed

        if model_spec.get("trend", False):
            trend_coef = pm.Normal("trend_coef", sigma=2)
            mu = mu + trend_coef*np.array(years)
        
        if model_spec.get("wind_power_law", False):
            wind_power = pm.Normal("wind_power", mu=3, sigma=2)
            mu = mu + np.log(wind_speed**wind_power)
        if model_spec.get("pressure_power_law", False):
            pressure_power = pm.Normal("pressure_power", mu=-2, sigma=2)
            mu = mu + np.log(pressure**pressure_power)
        
        if model_spec.get("tide_power_law", False):
            tide_power = pm.Normal("tide_power", mu=1, sigma=1)
            mu = mu + np.log(np.abs(tides)**tide_power + 1e-3) 
        if model_spec.get("travel_speed_power_law", False):
            travel_speed_power = pm.Normal("travel_speed_power", mu=-1, sigma=1)
            mu = mu + np.log(np.abs(travel_speed)**travel_speed_power + 1e-3)
        
        
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
        var_names.append("wind_speed_coef")
    if model_spec.get("travel_speed", False):
        var_names.append("travel_speed_coef")
    if model_spec.get("trend", False):
        var_names.append("trend_coef")
    if model_spec.get("wind_power_law", False):
        var_names.append("wind_power")
    if model_spec.get("pressure_power_law", False):
        var_names.append("pressure_power")
    if model_spec.get("tide_power_law", False):
        var_names.append("tide_power")
    if model_spec.get("travel_speed_power_law", False):
        var_names.append("travel_speed_power")
    

    
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
    plt.hist(ppc_values, bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title(f"Posterior Predictive Histogram ({model_name})")
    plt.legend()
    plt.savefig(os.path.join(model_path, f"{filename}_ppc_histogram.png"))
    plt.close()
    
    return trace, model_name


def hurricane_physical_model_APLR(df):
    path = r"./Speciale/Code/Week2/Plots"

    # Clean data
    df_clean = df[df['basedamage'] > 0].copy()
    df_clean = df_clean.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    df_clean["APLR"] = df_clean["basedamage"] / (df_clean["population"] * df_clean["WPC"])

    # Prepare variables
    BD = df_clean['basedamage'].values
    ATD = df_clean['ATD'].values
    population = df_clean['population'].values
    WPC = df_clean['WPC'].values
    wind_speed = df_clean['lf_wind'].values
    pressure = df_clean['lf_pressure'].values
    areas = ATD/BD*population*WPC
    tides = df_clean['Tide_Level'].values
    APLR = df_clean['APLR'].values
    category_1_wind_baseline = 74*0.868976242 #convert mph to knots
    category_1_pressure_baseline = 980 #mb
    wind_speed = wind_speed / category_1_wind_baseline
    pressure = pressure / category_1_pressure_baseline

    wind_ms = wind_speed * 0.514444 #convert knots to m/s
    with pm.Model() as model:
        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=7)
        pressure_coef_pos = pm.HalfNormal("pressure_coef_pos", sigma=2)
        pressure_coef = pm.Deterministic("pressure_coef", -pressure_coef_pos)#Doesnt add much, very correlated with wind_speed_coef
        tides_coef = pm.Normal("tides_coef",  sigma=0.5) #Doesnt add much
        b = pm.TruncatedNormal("b", mu=40.0, sigma=15.0, lower=1e-3)
        a = pm.TruncatedNormal("a", mu=10**(-3.5), sigma=1, lower=1e-8, upper=1e-2)
        c = pm.TruncatedNormal("c", mu=10.0, sigma=2.0, lower=0.1)

        mu = alpha + np.log(1/areas) + np.log(((wind_ms/b)**c + a)) +tides*tides_coef + pressure*pressure_coef
        obs = pm.LogNormal("obs", mu=mu, sigma=sigma, observed=APLR)
        
        trace = pm.sample(draws=2500, tune=1000, target_accept=0.95)
        summary = az.summary(trace, hdi_prob=0.95)
        print(summary)

    # Plot traces
    az.plot_trace(trace, var_names=["alpha", "sigma", "a", "b", "c", "tides_coef" ], figsize=(12, 12))
    plt.show()
    az.plot_pair(trace, var_names=["alpha", "sigma", "a", "b", "c", "tides_coef"], 
                 kind="kde", marginals=True, divergences=True)
    plt.show()

    # Posterior predictive check
    with model:
        ppc = pm.sample_posterior_predictive(trace)

    ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    # Histogram comparison
    combined = np.concatenate([np.log(APLR), np.log(ppc_values)])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(np.log(APLR)))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(np.log(APLR), bins=bins, density=True, alpha=0.5, label="Observed")
    plt.hist(np.log(ppc_values), bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title("Posterior Predictive Histogram")
    plt.legend()
    plt.show()
    
    return trace


def compare_models(df, model_specs, use_ND=False):
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
        trace, model_name = hurricane_physical_model(df, use_ND=use_ND, model_spec=spec)
        traces[model_name] = trace

    
    # Compare models using WAIC
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (WAIC)")
    print("="*80)
    comparison_df = az.compare(traces)
    print(comparison_df)
    #az.plot_compare(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(r"./Speciale/Code/Week2/Plots/model_comparison_waic.csv")
    
    # Also try LOO
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (LOO)")
    print("="*80)
    try:
        comparison_loo = az.compare(traces, ic="loo")
        az.plot_compare(comparison_loo)
        plt.savefig(r"./Speciale/Code/Week2/Plots/model_comparison_loo.png")
        plt.show()

        print(comparison_loo)
        comparison_loo.to_csv(r"./Speciale/Code/Week2/Plots/model_comparison_loo.csv")
    except Exception as e:
        print(f"LOO comparison failed (may need pareto_k adjustment): {e}")
    
    return comparison_df, traces


if __name__ == "__main__":
    #Example usage
    #df = pd.read_excel('./Speciale/Hurricane_data/Aslak_data.xls', sheet_name='ATD of ICAT', engine='xlrd')
    df = pd.read_csv('./Speciale/Hurricane_data/Aslak_data_with_tide_and_travelspeed.csv')
    df_clean = df.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    BD = np.array(df_clean['basedamage'].values)
    BD = BD[BD > 0]  #remove non-positive values

    plt.hist(np.log(BD), bins=20, density=True, alpha=0.6, color='g')
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title("Histogram of Base Damage")
    plt.savefig("./Speciale/Code/Week2/Plots/BaseDamage_histogram.png")
    plt.close()
    
    # Define model specifications to compare
    model_specs = [
        {"economic": True, "pressure": True, "tides": True, "wind": True, "travel_speed": True, "trend": True},   # Full 
        {"economic": True, "pressure": True, "tides": False, "wind": True, "travel_speed": True, "trend": False},
        {"economic": True, "pressure": True, "tides": True, "wind": False, "travel_speed": True, "trend": True, "wind_power_law": True},
        {"economic": True, "pressure": False, "tides": False, "wind": True, "travel_speed": True, "trend": True, "pressure_power_law": True},
        #{"economic": True, "pressure": True, "tides": True, "wind": True, "travel_speed": False, "trend": True, "travel_speed_power_law": True},
        {"economic": True, "pressure": True, "tides": False, "wind": True, "travel_speed": False, "trend": True, "tide_power_law": True},
        {"economic": True, "pressure": False, "tides": False, "wind": False, "travel_speed": True, "trend": True, "pressure_power_law": True, "wind_power_law": True},
       # {"economic": False, "pressure": False, "tides": False, "wind": False, "travel_speed": False, "trend": False, "wind_power_law": False, "pressure_power_law": False, "tide_power_law": False},        
    ]
    
    # Run comparison
    comparison_df, traces = compare_models(df, model_specs, use_ND=False)
    
    #hurricane_physical_model_APLR(df)
    #hurricane_physical_model(df, use_ND=True, model_spec=model_specs[0])


