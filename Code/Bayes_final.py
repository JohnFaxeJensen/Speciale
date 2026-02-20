import pymc as pm
import numpy as np
import arviz as az
from matplotlib import pyplot as plt
import os
import pandas as pd
import pytensor.tensor as pt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import sys
sys.path.append(r"./Speciale/Code")
from wind_eq_comparison_final import equation_11_wind
from ULLN import ulln_logp,  ulln_random
from Data_preprocessing.generate_csv import generate_csv_data

rng = np.random.default_rng(seed=42)

def estimate_upper_bound( pops, wpcs):
    exposure = pops * wpcs
    max_exposure = np.max(exposure)
    upper_bound = 10*max_exposure #set upper bound to 10 times max exposure
    return upper_bound


def add_wind_uncertainty(df):
    #calculate uncertainty in wind speed measurements based on period of record uncetainty here: chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.ncei.noaa.gov/sites/default/files/2025-04/IBTrACS_version4r01_Technical_Details.pdf
    get_uncertainty = lambda year: (
        30 if year < 1965 else
        20 if year < 1978 else
        15 if year < 1984 else
        10 if year < 2000 else
        7
    )
    wind_uncertainty_list = [get_uncertainty(year) for year in df['Year'].values]
    return wind_uncertainty_list

def get_pressure_uncertainty(wind_raw, wind_uncertainties): #uncertainty here: chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.nhc.noaa.gov/pdf/landsea-franklin-mwr2013.pdf
    #maybe add some kind of time dependency here as well
    #try to classify based on saffir-simpson categories
    #scale here: https://www.nhc.noaa.gov/aboutsshws.php

    pressure_uncertainty_list = []
    pressure_min_list = []
    pressure_max_list = []
    for wind, uncertainty in zip(wind_raw, wind_uncertainties):
        #estimate worst case uncertainty based on wind speed category
        wind_worst = wind + uncertainty
        if wind_worst < 64:  # Tropical Storm
            pressure_uncertainty = 2.8 # range (2-5)
            min_range = 2
            max_range = 5

        elif wind_worst <= 95:  # Category 1 + 2
            pressure_uncertainty = 3.5 # range(1.5–8)
            min_range = 1.5
            max_range = 8
        else:  # stronger storms
            pressure_uncertainty = 3.6 #range(1.5–10)
            min_range = 1.5
            max_range = 10
        pressure_uncertainty_list.append((pressure_uncertainty))
        pressure_min_list.append(min_range)
        pressure_max_list.append(max_range)

    return np.array(pressure_uncertainty_list), np.array(pressure_min_list), np.array(pressure_max_list)

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
    wind_uncertainties = add_wind_uncertainty(df_clean)
    wind_speed_raw = df_clean['lf_wind'].values
    pressure_raw = df_clean['lf_pressure'].values
    pressure_uncertainties, pressure_min_list, pressure_max_list = get_pressure_uncertainty(wind_speed_raw, wind_uncertainties) # comes as list of tuples (uncertainty, min_range, max_range) to allow for different distributions in pressure uncertainty if desired in future


    #convert wind and pressure measurements to draws to include measurement uncertainty

    area =10000 #value set by Aslak in study
    tides_m = df_clean['Tide_Level_lf'].values
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
    # modelled_wind_raw = equation_11_wind(pressure_raw, df_clean['lf_lat'].values, travel_speed)
    # # Orthogonalize residuals: wind_speed ~ modelled_wind, use residual of this regression
    # lr = LinearRegression().fit(modelled_wind_raw.reshape(-1, 1), wind_speed_raw)
    # wind_pred_raw = lr.predict(modelled_wind_raw.reshape(-1, 1))
    # residual_wind_raw = wind_speed_raw - wind_pred_raw
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
    model_name_rename_dict = {
        "economic": "ec",
        "pressure": "p",
        "wind": "w",
        "storm_tide": "st",
        "travel_speed": "ts",
        "trend": "tr",
        "pca": "pca",
        "inverse_barometer": "ib",
        "modelled_wind": "mw",
        "residual_wind": "rw",
        "wind_power_law": "wpl",
        "pressure_threshold": "pth",
        "storm_tide_threshold": "stth",
        "gc_hadisst": "gcH",
        "mdr_hadisst": "mdrH",
        "gc_icoads": "gcI",
        "mdr_icoads": "mdrI",
        "seasonal": "sea",
    }
    spec_parts = [model_name_rename_dict.get(part, part) for part in spec_parts]
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
    # modelled_wind_relative = modelled_wind_raw / category_1_wind_baseline
    # residual_wind_relative = residual_wind_raw / category_1_wind_baseline

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
    storm_surge_m = df_clean['Surge_m'].values
    has_storm_tide = ~np.isnan(storm_tides_m)
    has_surge = ~np.isnan(storm_surge_m)
    tidal_range_peak = df_clean['Tidal_Range_peak'].values
    #group by datum
    datum_idx = pd.Categorical(df_clean['Datum']).codes
    print(datum_idx)
    n_datums = len(np.unique(datum_idx))




    r34_mean = df_clean['R34_MEAN_RADIUS'].values
    r50_mean = df_clean['R50_MEAN_RADIUS'].values
    r64_mean = df_clean['R64_MEAN_RADIUS'].values

    economic_var = np.log(population*WPC/area)
    lr = LinearRegression().fit(economic_var.reshape(-1, 1), np.array(years))

    converted_msl_storm_tides = df_clean['Converted_MSL'].values
    converted_msl_uncertainty = df_clean['Converted_uncertainty_MSL'].values

    # Fill storm tide values: use converted if available, fallback to original measurement
    storm_tides_m_converted = np.where(
        ~np.isnan(converted_msl_storm_tides),  # condition: converted value exists
        converted_msl_storm_tides,             # use converted value
        storm_tides_m                          # fallback to original measurement
    )

    with pm.Model() as model:


        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=0, sigma=50)
        sigma = pm.HalfNormal("sigma", sigma=20)
        
        # Build mu dynamically based on model_spec
        mu = alpha
        
        for spec in model_spec:
            if 'wind' in spec: 
                wind_true = pm.Normal("wind_true", mu=wind_speed_raw,
                                        sigma=wind_uncertainties, shape=len(wind_speed_raw))
                wind_speed_relative = wind_true / category_1_wind_baseline
            if 'pressure' in spec:
                        #infer sigma from pressure uncertainty, but allow for some flexibility with a scaling factor
                pressure_sigma = pm.TruncatedNormal("pressure_sigma", 
                                                    mu=pressure_uncertainties, 
                                                    sigma=(pressure_max_list-pressure_min_list)/4, lower=pressure_min_list
                                                    , upper=pressure_max_list, shape=len(pressure_uncertainties))  
                
                pressure_true = pm.Normal("pressure_true", mu=pressure_raw,
                                        sigma=pressure_sigma, shape=len(pressure_raw)) # add sigma that is normal dist here

                delta_P = (1013.25-pressure_true)* 100  #convert to Pa
                delta_P_baseline = (1013.25 - category_1_pressure_baseline)*100
                delta_P_relative = delta_P / delta_P_baseline  #Pa

        #define water level as sum of tides and storm surge, and include uncertainty in both
            # if 'water_level' in spec:
                
            #     n = len(observed)
                
            #     water_level_imputed = np.where(has_storm_tide, storm_tides_m, storm_surge_m + 0.5*tidal_range_peak) #impute missing storm tide with sum of surge and average tide, to retain more information from tidal range
            #     water_level_true = pm.Normal("water_level_true", mu=water_level_imputed, sigma=tidal_range_peak*0.2, shape=n)

        if model_spec.get("raw_storm_tide", False):
            #just use surge as storm tide proxy

            storm_tides_m_imputed = np.where(has_storm_tide, storm_tides_m, storm_surge_m)

            storm_tide_coef = pm.Normal("storm_tide_coef", sigma=10)
            mu = mu + storm_tide_coef * storm_tides_m_imputed

        if model_spec.get("raw_surge", False):
            tide_mu = 0.5 * tidal_range_peak
            water_level_imputed = np.where(has_surge, storm_surge_m, storm_tides_m - tide_mu) #impute missing storm tide with sum of surge and average tide, to retain more information from tidal range
            storm_tide_plus_tide_coef = pm.Normal("storm_tide_plus_tide_coef", sigma=10)
            mu = mu + storm_tide_plus_tide_coef * water_level_imputed
        if model_spec.get("tidal_range_peak", False):
            tidal_range_coef = pm.Normal("tidal_range_coef", sigma=10)
            mu = mu + tidal_range_coef * tidal_range_peak

        if model_spec.get("water_level_converted", False):
            converted_msl_storm_tides = df['Converted_MSL'].values
            converted_msl_uncertainty = df['Converted_uncertainty_MSL'].values
            gauge_measurements = df['Manual check'] == 'gauge'
            #fill the converted values in storm tides :
            # Replace all storm tide values with converted values if they exist (not NaN)
            storm_tides_m_converted = np.where(
                ~np.isnan(converted_msl_storm_tides),  # condition: converted value exists
                converted_msl_storm_tides,             # use converted value
                storm_tides_m                          # fallback to original measurement
            )
            storm_tides_m_imputed = np.where(has_storm_tide, storm_tides_m_converted, storm_surge_m+ tidal_range_peak*0.5) 
            storm_tide_coef = pm.Normal("storm_tide_converted_coef", sigma=10)
            mu = mu + storm_tide_coef * storm_tides_m_imputed
        if model_spec.get("water_level_estimated", False):
            surge_mu = np.where(
                np.isnan(storm_surge_m),
                storm_tides_m_converted - 0.5 * tidal_range_peak,  # prior guess for missing surge
                storm_surge_m            # observed values
            )
            surge_sigmas = np.where(has_surge, 1.0, tidal_range_peak * 0.2)
            surge_true = pm.TruncatedNormal(
                "surge_true",
                mu=surge_mu,
                sigma=surge_sigmas,
                lower=0.0,
                shape=len(storm_surge_m)
            )
            tide_scale = tidal_range_peak * 0.3
            tide_offset = pm.Normal("tide_offset", mu=tidal_range_peak*0.5, sigma=tide_scale)
            storm_tide_true = surge_true + tide_offset
            pm.Normal(
                "storm_tide_like",
                mu=storm_tide_true[has_storm_tide],
                sigma=tide_scale[has_storm_tide],
                observed=storm_tides_m_converted[has_storm_tide]
            )
            pm.Normal(
                "surge_like",
                mu=surge_true[has_surge],
                sigma=surge_sigmas[has_surge],
                observed=storm_surge_m[has_surge]  # NaNs automatically handled as missing
            )
            beta_surge = pm.Normal("beta_surge", sigma=10)
            beta_tide = pm.Normal("beta_tide", sigma=10)
            #beta_storm_tide = pm.Normal("beta_storm_tide", sigma=10)
            mu = mu + beta_surge * surge_true + beta_tide * tide_offset #+ beta_storm_tide*storm_tide_true

        if model_spec.get("surge_estimated", False):
            surge_mu = np.where(
                np.isnan(storm_surge_m),
                storm_tides_m_converted - 0.5 * tidal_range_peak,  # prior guess for missing surge
                storm_surge_m            # observed values
            )
            surge_sigmas = np.where(has_surge, 1.0, tidal_range_peak * 0.2)
            surge_true = pm.TruncatedNormal(
                "surge_true",
                mu=surge_mu,
                sigma=surge_sigmas,
                lower=0.0,
                shape=len(storm_surge_m)
            )

            surge_like = pm.Normal(
                "surge_like",
                mu=surge_true[has_surge],
                sigma=surge_sigmas[has_surge],
                observed=storm_surge_m[has_surge]  # NaNs automatically handled as missing
            )
            surge_coef = pm.Normal("surge_coef", sigma=10)
            mu = mu + surge_coef * surge_true

        if model_spec.get("economic", False):
            economic_coef = pm.Normal("economic_coef", sigma=10)
            mu = mu + economic_coef*np.log(population*WPC/area)

        if model_spec.get("economic_exp", False):
            economic_coef_exp = pm.Normal("economic_coef_exp", sigma=10)
            mu = mu + economic_coef_exp*(population*WPC/area)


        if model_spec.get("economic_split", False):
            economic_coef_pop = pm.Normal("economic_coef_pop", sigma=10)
            economic_coef_wpc = pm.Normal("economic_coef_wpc", sigma=10)
            mu = mu + economic_coef_pop*np.log(population/area) + economic_coef_wpc*np.log(WPC)
        if model_spec.get("pressure", False):
            # Linear in log-space: log(damage) ~ coef * pressure
            pressure_coef = pm.Normal("pressure_coef", sigma=10)
            mu = mu + pressure_coef*delta_P_relative
        
        if model_spec.get("pressure_threshold", False):
            # Threshold response in log-space: only log(pressure) above threshold
            pressure_thresh = pm.Normal("pressure_threshold_val", mu=-1.0, sigma=0.5)
            pressure_thresh_coef = pm.Normal("pressure_thresh_coef", sigma=10)
            pressure_log = pm.math.log(delta_P_relative + 0.1)
            pressure_excess = pm.math.maximum(pressure_log - pressure_thresh, 0)
            mu = mu + pressure_thresh_coef * pressure_excess


        
        if model_spec.get("tides", False):
            tides_coef = pm.Normal("tides_coef", sigma=10)
            mu = mu + tides_m*tides_coef
        
        if model_spec.get("wind", False):
            wind_coef = pm.Normal("wind_coef", sigma=10)
            mu = mu + wind_coef*wind_speed_relative

        # if model_spec.get("modelled_wind", False):
        #     modelled_wind_coef = pm.Normal("modelled_wind_coef", sigma=5)
        #     mu = mu + modelled_wind_coef*modelled_wind_relative
        #     wind_coef_residual = pm.Normal("wind_coef_residual", sigma=5)
        #     mu = mu + wind_coef_residual*residual_wind_relative
        # if model_spec.get("residual_wind", False):
        #     wind_coef_residual = pm.Normal("wind_coef_residual", sigma=5)
        #     mu = mu + wind_coef_residual*residual_wind_relative
        if model_spec.get("wind_power_law", False):
            wind_power_law_coef = pm.Normal("wind_power_law_coef", mu=9, sigma=4)
            wind_power_law = np.log(wind_speed_relative)*(wind_power_law_coef)
            mu = mu + wind_power_law
        
        if model_spec.get("travel_speed", False):
            travel_speed_coef = pm.Normal("travel_speed_coef", sigma=10)
            mu = mu + travel_speed_coef*travel_speed

        if model_spec.get("trend", False):
            trend_coef = pm.Normal("trend_coef", mu=0, sigma=0.1)
            mu = mu + trend_coef*np.array(years)

        if model_spec.get("pca", False):
            pc1_coef = pm.Normal("pc1_coef", sigma=10)
            pc2_coef = pm.Normal("pc2_coef", sigma=10)
            mu = mu + pc1_coef*pc1 + pc2_coef*pc2

        if model_spec.get("inverse_barometer", False):
            ib_coef = pm.Normal("ib_coef", sigma=10)
            mu = mu + ib_coef*ib_vals_relative  # Example term for inverse barometer effect
        if model_spec.get("economic_raw", False):
            mu = mu + np.log(population*WPC/area)

        if model_spec.get("temp_anomaly_global", False):
            temp_anomaly_global_coef = pm.Normal("temp_anomaly_global_coef", sigma=10)
            mu = mu + temp_anomaly_global_coef*temp_anomaly_global
        if model_spec.get("seasonal", False):
            season_sin_coef = pm.Normal("season_sin_coef", sigma=10)
            season_cos_coef = pm.Normal("season_cos_coef", sigma=10)
            mu = mu + season_sin_coef * season_sin
            mu = mu + season_cos_coef * season_cos 
        if model_spec.get("pressure_trend_interaction", False):
            pressure_trend_interaction_coef = pm.HalfNormal("pressure_trend_interaction_coef", sigma=10)
            mu = mu + pressure_trend_interaction_coef * delta_P_relative * np.array(years)
        if model_spec.get("gc_hadisst", False):
            temp_coef = pm.Normal("coef_gc_hadisst", sigma=10)
            mu = mu + temp_coef * temp_hadisst_gc
        if model_spec.get("mdr_hadisst", False):
            temp_coef_mdr = pm.Normal("coef_mdr_hadisst", sigma=10)
            mu = mu + temp_coef_mdr * temp_hadisst_mdr
        if model_spec.get("gc_icoads", False):
            temp_coef = pm.Normal("coef_gc_icoads", sigma=10)
            mu = mu + temp_coef * temp_icoads_gc
        if model_spec.get("mdr_icoads", False):
            temp_coef_mdr = pm.Normal("coef_mdr_icoads", sigma=10)
            mu = mu + temp_coef_mdr * temp_icoads_mdr
        if model_spec.get("sea_level_rise", False):
            slope_sea_level = 0.0025  # meters per year
            sea_level_rise = slope_sea_level * np.array(years)
            sea_level_coef = pm.Normal("sea_level_coef", sigma=10)
            mu = mu + sea_level_coef * sea_level_rise
        if model_spec.get("ibtracks_speed", False):
            ibtracks_speed_coef = pm.Normal("ibtracks_speed_coef", sigma=10)
            mu = mu + ibtracks_speed_coef * (ibtracks_speed / 10)  # normalize to 10 m/s
        
        if model_spec.get("wind_vulnerability", False):
            v_threshold = pm.Normal("v_threshold", mu=50, sigma=10)
            v_half = pm.Normal("v_half", mu=120, sigma=15)
            v_coef = pm.Normal("v_coef", sigma=10)
            v_n = pm.math.maximum(wind_speed_raw-v_threshold,0)/(v_half-v_threshold)
            vulnerability = v_n**3/(1+v_n**3)
            mu = mu + v_coef*np.log(area*vulnerability + 1e-6)  # add small constant to avoid log(0)
        if model_spec.get('r34_mean', False):
            r34_coef = pm.Normal("r34_coef", sigma=10)
            mu = mu + r34_coef * r34_mean
        if model_spec.get('r50_mean', False):
            r50_coef = pm.Normal("r50_coef", sigma=10)
            mu = mu + r50_coef * r50_mean
        if model_spec.get('r64_mean', False):
            r64_coef = pm.Normal("r64_coef", sigma=10)
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
            
            state_vuln_coef = pm.Normal("state_vuln_coef", sigma=10)
            mu = mu + state_vuln_coef * state_vulnerability
        if model_spec.get("risk_score", False):
            risk_score_coef = pm.Normal("risk_score_coef", sigma=10)
            mu = mu + risk_score_coef * risk_scores





        if use_ulln:
            upper_min = np.max(np.exp(observed)) 
            print("ULLN upper min:", upper_min)
            upper_max = estimate_upper_bound( population, WPC) # the posterior is very dependent on prior and not well constrained by data
            print("ULLN upper max:", upper_max)
            upper = pm.Beta("upper", alpha=1.5, beta=3)* (upper_max - upper_min) + upper_min

            upper_deterministic = pm.Deterministic("upper_deterministic", upper)

            obs = pm.CustomDist("obs",
                                 mu, sigma, upper,
                                 logp=ulln_logp,
                                 random=ulln_random,
                                 observed=np.exp(observed)) #maybe comment on arimethic error here

        else:  
            obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)
        
        trace = pm.sample(draws=1000, tune=1000, target_accept=0.95, idata_kwargs={'log_likelihood':True})
        summary = az.summary(
            trace,
            hdi_prob=0.95,
        )
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
        var_names += ["upper_deterministic", "upper"]
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
    if model_spec.get("water_level_estimated", False):
        var_names += ["beta_surge", "beta_tide"]
        #var_names.append("beta_storm_tide")
    if model_spec.get("surge_estimated", False):
        var_names.append("surge_coef")




    

    

    

    
    axes = az.plot_trace(trace, var_names=var_names, figsize=(12, 12))
    fig = np.asarray(axes).ravel()[0].get_figure()

    fig.subplots_adjust(hspace=0.8, wspace=0.25)  # <- increase hspace for more vertical room
    fig.savefig(os.path.join(model_path, f"{filename}_trace.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    
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
    comparison_df = az.compare(traces, ic="loo", method="stacking", var_name="obs")  # Use LOO for comparison, but can also try WAIC
    print(comparison_df)
    #az.plot_compare(comparison_df)
    
    # Save comparison to CSV
    comparison_df.to_csv(r"./Speciale/Code/Simulations/model_comparison_waic.csv")
    
    # Also try LOO
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (LOO)")
    print("="*80)
    try:
        comparison_loo = az.compare(traces, ic="loo", method="stacking", var_name="obs")
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
    df_clean = df.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure', 'Lat_db', 'Lon_db'])
    print(df_clean.shape)
    df_clean = df_clean[df_clean['basedamage'] > 0]
    print(df_clean.shape)
    df_clean = df_clean[df_clean['ND'] > 0]
    print(df_clean.shape)
    #try data past 1960 only
    #df_clean = df_clean[df_clean['Year'] >= 1960]


    model_specs = [
        {"economic": True, "pressure": True, 'trend': True, 'surge_estimated': True},
        {"economic": True, "pressure": True, 'trend': True, 'water_level_estimated': True},


        #{"economic": True, "pressure": True, "water_level": True, "trend": True},
        # {"economic": True, "pressure": True},
        # {"economic": True, "storm_tide": True},
        # {"economic": True, "storm_tide": True, 'trend': True},
        # {"economic": True, "wind": True, "storm_tide": True},
        # {"economic": True, "wind": True, "storm_tide": True, 'trend': True},
        # {"economic": True, "modelled_wind": True, "storm_tide": True, 'trend': True},
  

    ]
    #hurricane_physical_model(df_clean, model_spec={"trend": True,  "water_level_estimated": True, "pressure": True, "economic": True, "wind": True},use_ulln=True, observed_variable='basedamage')
    comparison_df, traces = compare_models(df_clean, model_specs, use_ulln=True, observed_variable='basedamage')
        # Compare both methods

