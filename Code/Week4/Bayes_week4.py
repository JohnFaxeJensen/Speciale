import pymc as pm
import numpy as np
import scipy.stats as stats
import arviz as az
from matplotlib import pyplot as plt
import os
import pandas as pd
from pytensor.printing import Print

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from wind_eq_comparison import equation_11_wind

#Difference between this and the week 2 is that i try to include new model terms
#such as inverse barometer effect and using a model to transform pressure to wind speed

def hurricane_physical_model(df,  model_spec=None, ATD=False):
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
    basedamage = df_clean['basedamage'].values
    if ATD:
        observed = np.log(df_clean['ATD'].values)
    else:
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
    travel_speed = df_clean['travel_speed_after_landfall_m_s'].values
    travel_speed_before = df_clean['travel_speed_before_landfall_m_s'].values
    modelled_wind_raw = equation_11_wind(pressure_raw, df_clean['lf_lat'].values, travel_speed)
    # Orthogonalize residuals: wind_speed ~ modelled_wind, use residual of this regression
    lr = LinearRegression().fit(modelled_wind_raw.reshape(-1, 1), wind_speed_raw)
    wind_pred_raw = lr.predict(modelled_wind_raw.reshape(-1, 1))
    residual_wind_raw = wind_speed_raw - wind_pred_raw

    # Create model name based on spec
    spec_parts = [k for k, v in model_spec.items() if v]
    model_name = "_".join(spec_parts)
    if ATD:
        filename = f"ATD_hurricane_model_{model_name}"
    else:
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
    with pm.Model() as model:
        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=17, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=5)
        
        # Build mu dynamically based on model_spec
        mu = alpha
        
        if model_spec.get("economic", False):
            economic_coef = pm.Normal("economic_coef", sigma=3)
            if ATD:
                mu = mu + economic_coef*np.log(population*WPC/area)
            else:
                mu = mu + economic_coef*np.log(population*WPC/area)
        
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
        
        if model_spec.get("travel_speed", False):
            travel_speed_coef = pm.Normal("travel_speed_coef", sigma=3)
            mu = mu + travel_speed_coef*travel_speed

        if model_spec.get("trend", False):
            trend_coef = pm.Normal("trend_coef", sigma=3)
            mu = mu + trend_coef*np.array(years)
        if model_spec.get("pca", False):
            pc1_coef = pm.Normal("pc1_coef", sigma=3)
            pc2_coef = pm.Normal("pc2_coef", sigma=3)
            mu = mu + pc1_coef*pc1 + pc2_coef*pc2
        if model_spec.get("multi_variate", False):
            data = np.array([wind_speed_relative , pressure_relative])
            covariance_matrix = np.cov(data)
            mv_coef = pm.MvNormal("mv_coef", mu=np.zeros(2), cov=covariance_matrix, shape=2)
            mu = mu + mv_coef[0]*wind_speed_relative + mv_coef[1]*pressure_relative
        if model_spec.get("inverse_barometer", False):
            ib_coef = pm.Normal("ib_coef", sigma=3)
            mu = mu + ib_coef*ib_vals_relative  # Example term for inverse barometer effect
        if model_spec.get("economic_raw", False):
            if ATD:
                mu = mu + np.log((population*WPC/area))
            else:
                mu = mu + np.log(population*WPC/area)
        if model_spec.get("water_level", False):
            water_level_coef = pm.Normal("water_level_coef", sigma=3)
            water_level = tides_m + ib_vals_raw
            mu = mu + water_level_coef*water_level



        if ATD:
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
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
    if model_spec.get("multi_variate", False):
        var_names.append("mv_coef")
    if model_spec.get("inverse_barometer", False):
        var_names.append("ib_coef")
    if model_spec.get("water_level", False):
        var_names.append("water_level_coef")
    if model_spec.get("residual_wind", False):
        var_names.append("wind_coef_residual")

    

    

    
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





def compare_models(df, model_specs, ATD=False):
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
        trace, model_name = hurricane_physical_model(df, model_spec=spec, ATD=ATD)
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



if __name__ == "__main__":
    #Example usage
    #df = pd.read_excel('./Speciale/Hurricane_data/Aslak_data.xls', sheet_name='ATD of ICAT', engine='xlrd')
    df = pd.read_csv('./Speciale/Hurricane_data/Aslak_data_with_tide_and_travelspeed.csv')
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
        {"economic": True, "trend": True, "wind": True},
        # {"economic": True, "pressure": True, "trend": True, "inverse_barometer": True},
        {"economic": True, "pressure": True, "trend": True},
        {"economic": True, "pressure": True, "trend": True, "wind": True},
        {"economic": True, "modelled_wind": True, "trend": True},
        {"economic": True, "modelled_wind": True, "trend": True, "inverse_barometer": True},
        {"economic": True,"pressure": True, "trend": True, "travel_speed": True},
        {"economic": True,"pressure": True, "trend": True, "tides": True},
        {"economic": True,"pressure": True, "trend": True, "inverse_barometer": True},

        # {"economic": True, "pressure": True, "trend": True, "residual_wind": True},
        # {"economic": True, "pressure": True, "trend": True, "travel_speed": True},

    ]
    # model_specs_massive = [
    #     # Baselines
    #     {"economic": True},
    #     {"economic": True, "trend": True},
    #     {"raw_economic": True, "trend": True},
    #     {"economic": True, "inverse_barometer": True},
    #     {"economic": True, "trend": True, "inverse_barometer": True},

    #     # Single-physics with econ
    #     {"economic": True, "wind": True},
    #     {"economic": True, "pressure": True},
    #     {"economic": True, "tides": True},
    #     {"economic": True, "travel_speed": True},
    #     {"economic": True, "modelled_wind": True},
    #     {"economic": True, "water_level": True},

    #     # Pairs with econ
    #     {"economic": True, "wind": True, "pressure": True},
    #     {"economic": True, "wind": True, "travel_speed": True},
    #     {"economic": True, "pressure": True, "inverse_barometer": True},
    #     {"economic": True, "tides": True, "inverse_barometer": True},
    #     {"economic": True, "modelled_wind": True, "pressure": True},
    #     {"economic": True, "modelled_wind": True, "travel_speed": True},
    #     {"economic": True, "water_level": True, "trend": True},
    #     {"economic": True, "tides": True, "trend": True},

    #     # Trios with econ
    #     {"economic": True, "wind": True, "pressure": True, "trend": True},
    #     {"economic": True, "wind": True, "pressure": True, "travel_speed": True},
    #     {"economic": True, "pressure": True, "tides": True, "trend": True},
    #     {"economic": True, "modelled_wind": True, "pressure": True, "trend": True},
    #     {"economic": True, "modelled_wind": True, "travel_speed": True, "trend": True},
    #     {"economic": True, "wind": True, "inverse_barometer": True, "trend": True},

    #     # “Full-ish” (no mixing wind with modelled_wind; water_level is exclusive)
    #     {"economic": True, "wind": True, "pressure": True, "travel_speed": True, "trend": True, "inverse_barometer": True},
    #     {"economic": True, "modelled_wind": True, "pressure": True, "travel_speed": True, "trend": True, "inverse_barometer": True},
    #     {"economic": True, "water_level": True, "pressure": True, "travel_speed": True, "trend": True},

    #     # PCA / Multivariate alternatives (with econ)
    #     {"economic": True, "pca": True},
    #     {"economic": True, "pca": True, "trend": True},
    #     {"economic": True, "pca": True, "travel_speed": True},
    #     {"economic": True, "multi_variate": True},
    #     {"economic": True, "multi_variate": True, "trend": True},
    #     {"economic": True, "multi_variate": True, "travel_speed": True},

    #     # Physics-only (for reference)
    #     {"wind": True, "pressure": True},
    #     {"modelled_wind": True, "pressure": True},
    #     {"water_level": True, "trend": True},
    #     {"wind": True, "pressure": True, "trend": True},
    #   ]
    
    # Run comparison
    comparison_df, traces = compare_models(df, model_specs, ATD=False)
    
    #hurricane_physical_model_APLR(df)
    #hurricane_physical_model(df, model_spec=model_specs[0])


