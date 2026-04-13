import pymc as pm
import numpy as np
import arviz as az
import arviz_stats as az_stats
from matplotlib import pyplot as plt
import os
import pandas as pd
import pytensor
from pymc_bart import BART

pytensor.config.cxx=""

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
import sys
sys.path.append(r"./Speciale/Code")
#from wind_eq_comparison_final import equation_11_wind
from ULLN import ulln_logp,  ulln_random
from LogStudentT import log_studentt_logp, log_studentt_random
#from Data_preprocessing.generate_csv import generate_csv_data

rng = np.random.default_rng(seed=42)

# ============================================================================
# PREDEFINED MODEL SPECIFICATIONS (Based on BART discoveries)
# ============================================================================



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

def get_pressure_uncertainty(wind_raw, wind_uncertainties, years):
     #uncertainty here: chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.nhc.noaa.gov/pdf/landsea-franklin-mwr2013.pdf
    #maybe add some kind of time dependency here as well
    #try to classify based on saffir-simpson categories
    #scale here: https://www.nhc.noaa.gov/aboutsshws.php
    def get_time_factor(year):
        if year < 1965:
            return 2 # this should probably change, just an estimate I made to reflect higher uncertainty in older measurements, but this is a very rough estimate and could be improved with more research into the historical measurement techniques and their uncertainties
        else:
            return 1.0
    pressure_uncertainty_list = []
    pressure_min_list = []
    pressure_max_list = []
    for wind, uncertainty, year in zip(wind_raw, wind_uncertainties, years):
        #estimate worst case uncertainty based on wind speed category
        time_factor = get_time_factor(year)
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
        pressure_uncertainty_list.append((pressure_uncertainty*time_factor))
        pressure_min_list.append(min_range*time_factor)
        pressure_max_list.append(max_range*time_factor)

    return np.array(pressure_uncertainty_list), np.array(pressure_min_list), np.array(pressure_max_list)


def hurricane_physical_model(df, model_spec=None, observed_variable='basedamage'):
    """
    Build and fit a hurricane damage model with configurable terms and likelihood.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    model_spec : dict, optional
        Specify which terms to include plus likelihood choice. If None, uses full model.
        
        Feature specification example:
            {"exposure": True, "pressure": True, "wind": True, "trend": False, ...}
        
        Likelihood specification:
            {"likelihood": "lognormal"}  # Options: "lognormal", "log_studentt", "ulln"
        
        Full example:
            {
                "exposure": True, 
                "pressure": True, 
                "wind": True,
                "likelihood": "log_studentt"
            }
        
        Default likelihood: "lognormal"
    
    observed_variable : str, optional
        Which variable to use as observation (default: 'basedamage')
        Options: 'basedamage', 'ATD', 'ND'
    
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
    
    # Extract likelihood from model_spec (default: lognormal)
    likelihood = model_spec.get("likelihood", "lognormal")
    valid_likelihoods = ["lognormal", "log_studentt", "ulln", "pareto"]
    if likelihood not in valid_likelihoods:
        raise ValueError(f"likelihood must be one of {valid_likelihoods}, got {likelihood}")
    df_clean = df.dropna(subset=[observed_variable, 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    df_clean = df_clean[df_clean[observed_variable] > 0].copy()
    if "surge_full" in model_spec:
        df_clean = df_clean.dropna(subset=['Surge_m_full_conversion_AN_as_surge']) # need at least surge or storm tide measurement to estimate surge
    # Clean data
    observed = df_clean[observed_variable].values
    # transform to log-space
    observed = np.log(observed)

    area =10000 #value set by Aslak in study
    tides_m = df_clean['Tide_Level_lf'].values
    timestamps = df_clean['lf_ISO_TIME'].values 
    years = pd.to_datetime(timestamps).year
    delta_years = [int(year - int(years[0])) for year in years]

    population = df_clean['population'].values
    WPC = df_clean['WPC'].values
    wind_uncertainties = add_wind_uncertainty(df_clean)
    wind_speed_raw = df_clean['lf_wind'].values
    pressure_raw = df_clean['lf_pressure'].values
    pressure_uncertainties, pressure_min_list, pressure_max_list = get_pressure_uncertainty(wind_speed_raw, wind_uncertainties, years) # comes as list of tuples (uncertainty, min_range, max_range) to allow for different distributions in pressure uncertainty if desired in future


    #convert wind and pressure measurements to draws to include measurement uncertainty

 
    # Seasonal (day-of-year) features
    doy = pd.to_datetime(timestamps).dayofyear.values.astype(float)
    theta = 2 * np.pi * (doy / 365.25)
    season_sin = np.sin(theta)
    season_cos = np.cos(theta)

    travel_speed = df_clean['lf_speed_after_ms'].values
    travel_speed_before = df_clean['lf_speed_before_ms'].values

    # load and detrend anomaly 
    temp_anomaly_global = df_clean['Anomaly_global'].values
    temp_anomaly_global = temp_anomaly_global - (np.array(delta_years))*0.008 #rough detrend
    # modelled_wind_raw = equation_11_wind(pressure_raw, df_clean['lf_lat'].values, travel_speed)
    # # Orthogonalize residuals: wind_speed ~ modelled_wind, use residual of this regression
    # lr = LinearRegression().fit(modelled_wind_raw.reshape(-1, 1), wind_speed_raw)
    # wind_pred_raw = lr.predict(modelled_wind_raw.reshape(-1, 1))
    # residual_wind_raw = wind_speed_raw - wind_pred_raw
    temp_hadisst_gc = df_clean['mean_temp_hadisst_gc'].values
    temp_hadisst_mdr = df_clean['mean_temp_hadisst_mdr'].values
    temp_icoads_gc = df_clean['mean_temp_icoads_gc'].values
    temp_icoads_mdr = df_clean['mean_temp_icoads_mdr'].values
    baseline_decade = (np.array(delta_years) >= 70) & (np.array(delta_years) < 80) 
    temp_hadisst_gc_anom = temp_hadisst_gc - temp_hadisst_gc[baseline_decade].mean()
    temp_hadisst_mdr_anom = temp_hadisst_mdr - temp_hadisst_mdr[baseline_decade].mean()
    temp_icoads_gc_anom = temp_icoads_gc - temp_icoads_gc[baseline_decade].mean()
    temp_icoads_mdr_anom = temp_icoads_mdr - temp_icoads_mdr[baseline_decade].mean()
    risk_scores = df_clean['Hurricane_Risk_Score'].values



    # Create model name based on spec
    spec_parts = [k for k, v in model_spec.items() if v and k != "likelihood"]
    
    # Likelihood abbreviations
    likelihood_abbrev = {
        "lognormal": "ln",
        "log_studentt": "lst",
        "pareto": "pw",
        "ulln": "ulln",
    }
    likelihood_name = likelihood_abbrev.get(likelihood, "ln")  # default to "ln" if not found
    
    model_name_rename_dict = {
        # exposure features
        "exposure": "ec",
        "exposure_exp": "ec_exp",
        "exposure_split": "ec_spl",
        "exposure_vulnerability": "ec_vuln",
        "exposure_raw": "ec_raw",
        "exposure_trend_interaction": "ec_tr_int",
        # Pressure features
        "pressure": "p",
        "pressure_linear": "p_lin",
        "pressure_quadratic": "p_quad",
        "pressure_cubic": "p_cub",
        "pressure_poly_2": "p_p2",
        "pressure_poly_3": "p_p3",
        "pressure_threshold": "p_th",
        "pressure_trend_interaction": "p_tr_int",
        "raw_pressure": "rp",
        # Wind features
        "wind": "w",
        "wind_power_law": "wpl",
        "wind_expert_exp": "wexp",
        "raw_wind": "rw",
        "wind_vulnerability": "w_vuln",
        "wind_pressure_ratio": "wpr",
        "residual_wind": "res_w",
        "modelled_wind": "mw",
        # Surge/tide features
        "surge_full": "surg_f",
        "surge_full_trend_interaction": "surg_f_tr_int",
        "surge_AN_uncertainty": "surg_an",
        "surge_small_conversion": "surg_sm",
        "storm_tide_as_surge": "st_surg",
        "surge": "surg",
        "tides": "tid",
        # Temporal features
        "trend_linear": "tr_lin",
        "trend_quadratic": "tr_quad",
        "travel_speed": "ts",
        "trend": "tr",
        "sea_level_rise": "slr",
        # Atmospheric/environmental
        "inverse_barometer": "ib",
        "seasonal": "sea",
        "temp_anomaly_global": "tag",
        "ibtracks_speed": "ibts",
        # Temperature features
        "gc_hadisst": "gcH",
        "mdr_hadisst": "mdrH",
        "gc_icoads": "gcI",
        "mdr_icoads": "mdrI",
        # Storm size features
        "r34_mean": "r34",
        "r50_mean": "r50",
        "r64_mean": "r64",
        # Vulnerability/exposure features
        "vulnerability": "vuln",
        "risk_score": "rs",
        "fitted_exposure": "fe",
        # Interactions
        "interaction_pressure_time": "p_t_int",
        "interaction_pressure_exposure": "p_e_int",
        # Other
        "pca": "pca",
        "storm_tide": "st",
        "storm_tide_threshold": "stth",
    }
    spec_parts = [model_name_rename_dict.get(part, part) for part in spec_parts]
    model_name = "_".join(spec_parts)
    # Add likelihood suffix (e.g., "ec_p_tr_surg_f_ln" for lognormal)
    model_name = f"{model_name}_{likelihood_name}"
    filename = f"{observed_variable}_hurricane_model_{model_name}"
    model_path = os.path.join(r"./Speciale/Code/Simulations/Plots", filename)
    os.makedirs(model_path, exist_ok=True)

    category_1_wind_baseline = 95*0.868976242 #convert mph to knots
    category_1_pressure_baseline = 980 #mb

    travel_speed = travel_speed / 10 #normalize to 10 m/s
    travel_speed_before = travel_speed_before / 10 #normalize to 10 m/s
    # modelled_wind_relative = modelled_wind_raw / category_1_wind_baseline
    # residual_wind_relative = residual_wind_raw / category_1_wind_baseline



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
    






    r34_mean = df_clean['R34_MEAN_RADIUS'].values
    r50_mean = df_clean['R50_MEAN_RADIUS'].values
    r64_mean = df_clean['R64_MEAN_RADIUS'].values




    with pm.Model() as model:
        var_names = []
        # Priors for the linear combination

        # Noise hierarchy (independent)
        sigma = pm.HalfNormal("sigma", sigma=20)
        sigma_total = sigma
        var_names.extend(["sigma"])
        # Build mu dynamically based on model_spec
        mu = 0 

        wind_true_specs = ["wind",]
        needs_wind_true = any(model_spec.get(spec, False) for spec in wind_true_specs)
        pressure_true_specs = ["pressure", "pressure_linear", "pressure_quadratic", "pressure_cubic", "pressure_threshold", "pressure_trend_interaction", "pressure_poly_2", "pressure_poly_3", "pressure_poly_4", "pressure_poly_trend", "interaction_pressure_time", "interaction_pressure_exposure", 'pressure_bart', 'pressure_bart3']
        need_pressure_true = any(model_spec.get(spec, False) for spec in pressure_true_specs)
        if needs_wind_true:
            wind_true = pm.Normal("wind_true", 
                                mu=wind_speed_raw, 
                                sigma=wind_uncertainties, 
                                shape=len(wind_speed_raw))
            wind_speed_relative = wind_true / category_1_wind_baseline
            
        if need_pressure_true:

            pressure_sigma = pm.TruncatedNormal("pressure_sigma", 
                                                mu=pressure_uncertainties, 
                                                sigma=(pressure_max_list-pressure_min_list)/4, lower=pressure_min_list
                                                , upper=pressure_max_list, shape=len(pressure_uncertainties))
            # pressure_sigma = pm.Uniform("pressure_sigma",
            #                 lower=pressure_min_list,
            #                 upper=pressure_max_list,
            #                 shape=len(pressure_uncertainties))

            pressure_true = pm.Normal("pressure_true", mu=pressure_raw,
                                    sigma=pressure_sigma, shape=len(pressure_raw)) # maybe change this here
            # pm.Normal("pressure_like",        #maybe include the likelihood prob not.
            # mu=pressure_true,
            # sigma=pressure_sigma,  # each obs constrains proportional to confidence
            # observed=pressure_raw)

            delta_P = (1013.25-pressure_true)* 100  #convert to Pa
            delta_P_baseline = (1013.25 - category_1_pressure_baseline)*100
            delta_P_relative = delta_P / delta_P_baseline  #Pa
        if model_spec.get("sigma_pressure", False):
            # Standardize pressure_sigma to avoid scale issues
            pressure_sigma_standardized = (pressure_sigma - pressure_sigma.mean()) / (pressure_sigma.std() + 1e-8)
            # Use exponentiated form for numerical stability and positivity
            pressure_sigma_coef = pm.Normal("pressure_sigma_coef", mu=0, sigma=0.1)
            # sigma_multiplier = exp(coef * standardized_sigma) ensures sigma stays positive
            sigma_multiplier = pm.math.exp(pressure_sigma_coef * pressure_sigma_standardized)
            sigma_total = sigma * sigma_multiplier
            var_names.append("pressure_sigma_coef")
        if model_spec.get("alpha", False):
            alpha = pm.Normal("alpha", mu=0, sigma=20)
            mu += alpha
            var_names.append("alpha")
        if model_spec.get("alpha_by_intensity", False):
            #create hierarchical alpha by category of storm wind-scale (based on Saffir-Simpson scale, but using 4 categories: Tropical Storm, Cat 1, Cat 2-3, Cat 4-5)
            wind_speed_bins = [0, 64, 95, 130, np.inf] # in knots
            wind_speed_categories = np.digitize(wind_speed_raw, bins=wind_speed_bins) - 1  # categories will be 0, 1, 2, 3
            n_categories = len(wind_speed_bins) - 1  # Always 4 categories
            
            # Hyperpriors for the hierarchy (group-level variation)
            mu_alpha_intensity = pm.Normal("mu_alpha_intensity", mu=0, sigma=20)
            sigma_alpha_intensity = pm.HalfNormal("sigma_alpha_intensity", sigma=5)
            
            # Category-level alphas drawn from the hyperprior (allows shrinkage)
            alpha_by_category = pm.Normal("alpha_by_category", mu=mu_alpha_intensity, sigma=sigma_alpha_intensity, shape=n_categories)
            alpha_by_obs = alpha_by_category[wind_speed_categories]
            mu += alpha_by_obs
            
            var_names.extend(["mu_alpha_intensity", "sigma_alpha_intensity", "alpha_by_category"])
        if model_spec.get("sigma_exposure", False):
            exposure = population * WPC
            exposure_coef_sigma = pm.Normal("exposure_coef_sigma", sigma=5)
            sigma_total = sigma_total + exposure_coef_sigma*np.log(exposure/10000)
            var_names.append("exposure_coef_sigma")

        if model_spec.get("raw_pressure", False):
            pressure_coef = pm.Normal("pressure_coef", sigma=10)
            delta_P = (1013.25-pressure_raw)* 100  #convert to Pa
            delta_P_baseline = (1013.25 - category_1_pressure_baseline)*100
            delta_P_relative = delta_P / delta_P_baseline  #Pa
            mu = mu + pressure_coef*delta_P_relative
        if model_spec.get("raw_wind", False):
            wind_coef = pm.Normal("wind_coef", sigma=5)
            wind_speed_relative = wind_speed_raw / category_1_wind_baseline
            mu = mu + wind_coef*wind_speed_relative
        if model_spec.get("wind_pressure_ratio", False):
            wind_pressure_coef = pm.Normal("wind_pressure_coef", sigma=5)
            wind_pressure_ratio = (wind_speed_raw / (pressure_raw + 1)) / (category_1_wind_baseline / (category_1_pressure_baseline + 1))
            mu = mu + wind_pressure_coef*wind_pressure_ratio
        


        #     tidal_diff_percentage = np.array([0,0,0.023448276,0.255715495,0.100917431,0,0,0,0.531958763,0.161637931,0.050701187,0.131818182,0.02739726,0.203870968]) #from excel file

        #     # Calculate statistics

        #     std_diff = np.std(tidal_diff_percentage)
        #     surge_mu = np.where(
        #         np.isnan(storm_surge_m),
        #         storm_tides_m_converted - 0.5 * tidal_range_peak,  # prior guess for missing surge
        #         storm_surge_m            # observed values
        #     )
            


        #     # Empirical tidal timing uncertainty from 14-station analysis (std dev of measured offsets)
        #     empirical_tidal_std = std_diff*storm_tides_m_converted  #multiply total water level with std of percentage difference to get empirical tidal uncertainty in meters
        #     measurement_uncertainty = np.array([get_tide_instrument_error(year) for year in years])  #uncertainty from tide gauge measurement, based on era and technology)
        #     # add extra uncertainty from unknown datum
        #     # make a full-length boolean array for unknown datum per observation
        #     has_unknown_datum = (df_clean['Datum'].values == 'Unknown')
        #     datum_conversion_uncertainty = converted_msl_uncertainty
        #     datum_uncertainty = tidal_range_peak*0.33
        #     # maybe add location uncertainty as well
            
        #     # For missing surge observations, include tidal + measurement uncertainty,
        #     # and add datum uncertainty only where the datum is unknown.
        #     storm_tide_error = (
        #         empirical_tidal_std**2 + measurement_uncertainty**2 + np.where(has_unknown_datum, datum_uncertainty**2, 0.0) + np.where(~np.isnan(converted_msl_storm_tides), datum_conversion_uncertainty**2, 0.0)
        #     )
        #     storm_tide_error = np.sqrt(storm_tide_error)

        #     surge_prior_sigma = np.where(
        #         has_surge,
        #         measurement_uncertainty,  # almost deterministic for observed
        #         storm_tide_error
        #     )


        #     # surge_true = pm.TruncatedNormal(
        #     #     "surge_true",
        #     #     mu=surge_mu,
        #     #     sigma=surge_prior_sigma,
        #     #     lower=0.0,
        #     #     shape=len(storm_surge_m)
        #     # )
        #     sigma_log_raw = surge_prior_sigma / np.maximum(surge_mu, 1e-3)
        #     sigma_log = np.clip(sigma_log_raw, 0.05, 1.0)
        #     surge_true = pm.LogNormal(
        #         "surge_true",
        #         mu=np.log(np.maximum(surge_mu, 1e-6)),  # add small constant to avoid log(0)
        #         sigma=sigma_log,
        #         shape=len(storm_surge_m)
        #     )

        #     surge_coef = pm.Normal("surge_coef", sigma=10)
        #     mu = mu + surge_coef * surge_true
        

        if model_spec.get("surge_full", False):
            surge_mu = df_clean['Surge_m_full_conversion_AN_as_surge'].values
            surge_uncertainty = df_clean['Uncertainty_m_full_conversion_AN_as_surge'].values
            surge_true = pm.TruncatedNormal(
                "surge_true",
                mu=surge_mu,
                sigma=surge_uncertainty,
                lower=0.0,
                shape=len(surge_mu)
            )

            surge_coef = pm.Normal("surge_coef", sigma=10)
            mu = mu + surge_coef * surge_true
            var_names.append("surge_coef")

        if model_spec.get("surge_AN_uncertainty", False):
            surge_mu = df_clean['Surge_m_full_conversion_AN_as_unknown'].values
            surge_uncertainty = df_clean['Uncertainty_m_full_conversion_AN_as_unknown'].values
            surge_true = pm.TruncatedNormal(
                "surge_true_AN",
                mu=surge_mu,
                sigma=surge_uncertainty,
                lower=0.0,
                shape=len(surge_mu)
            )
            surge_coef_AN = pm.Normal("surge_coef_AN", sigma=10)
            mu = mu + surge_coef_AN * surge_true
            var_names.append("surge_coef_AN")
        if model_spec.get("surge_small_conversion", False):
            surge_mu = df_clean['Surge_m_subtract_peak_tide'].values
            surge_uncertainty = df_clean['Uncertainty_m_subtract_peak_tide'].values
            surge_true = pm.TruncatedNormal(
                "surge_true_small",
                mu=surge_mu,
                sigma=surge_uncertainty,
                lower=0.0,
                shape=len(surge_mu)
            )
            surge_coef_small = pm.Normal("surge_coef_small", sigma=10)
            mu = mu + surge_coef_small * surge_true
            var_names.append("surge_coef_small")
        
        if model_spec.get("storm_tide_as_surge", False):
            surge_mu = df_clean['Surge_m_storm_tide_as_surge'].values
            surge_uncertainty = df_clean['Uncertainty_m_storm_tide_as_surge'].values
            surge_true = pm.TruncatedNormal(
                "surge_true_storm_tide",
                mu=surge_mu,
                sigma=surge_uncertainty,
                lower=0.0,
                shape=len(surge_mu)
            )
            surge_coef_storm_tide = pm.Normal("surge_coef_storm_tide", sigma=10)
            mu = mu + surge_coef_storm_tide * surge_true
            var_names.append("surge_coef_storm_tide")

        if model_spec.get("surge_full_trend_interaction", False):
            surge_mu = df_clean['Surge_m_full_conversion_AN_as_surge'].values
            surge_uncertainty = df_clean['Uncertainty_m_full_conversion_AN_as_surge'].values
            surge_true = pm.TruncatedNormal(
                "surge_true",
                mu=surge_mu,
                sigma=surge_uncertainty,
                lower=0.0,
                shape=len(surge_mu)
            )
            surge_full_trend_interaction_coef = pm.Normal("surge_full_trend_interaction_coef", sigma=10)
            mu = mu + surge_full_trend_interaction_coef * surge_mu * np.array(delta_years)
            var_names.append("surge_full_trend_interaction_coef")

        if model_spec.get("exposure", False):
            exposure_coef = pm.Normal("exposure_coef", sigma=10)
            mu = mu + exposure_coef*np.log(population*WPC/area)
            var_names.append("exposure_coef")
        if model_spec.get("exposure_poly", False):
            exposure_coef_poly1 = pm.Normal("exposure_coef_poly1", sigma=10)
            exposure_coef_poly2 = pm.Normal("exposure_coef_poly2", sigma=10)
            mu = mu + exposure_coef_poly1*np.log(population*WPC/area) + exposure_coef_poly2*(np.log(population*WPC/area))**2
            var_names.extend(["exposure_coef_poly1", "exposure_coef_poly2"])

        if model_spec.get("exposure_exp", False):
            exposure_coef_exp = pm.Normal("exposure_coef_exp", sigma=10)
            mu = mu + exposure_coef_exp*(population*WPC/area)
            var_names.append("exposure_coef_exp")
        if model_spec.get("exposure_trend_interaction", False):
            exposure_trend_interaction_coef = pm.Normal("exposure_trend_interaction_coef", sigma=10)
            mu = mu + exposure_trend_interaction_coef * np.log(population*WPC/area) * np.array(delta_years)
            var_names.append("exposure_trend_interaction_coef")
        if model_spec.get("exposure_vulnerability", False):
            unique_states = df_clean['lf_state'].unique()
            # Create mapping from state names to integer indices
            state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
            state_indices = np.array([state_to_idx[state] for state in df_clean['lf_state'].values])
            
            vulnerability_by_state = pm.Normal("vulnerability_by_state", mu=1, sigma=0.3, shape=len(unique_states))
            vulnerability_by_obs = vulnerability_by_state[state_indices]
            vulnerability_coef = pm.Normal("vulnerability_coef", sigma=10)
            exposure_coef_vuln = pm.Normal("exposure_coef_vuln", sigma=10)
            mu = mu + exposure_coef_vuln*np.log(population*WPC/area) + vulnerability_coef*vulnerability_by_obs
            var_names.extend(["exposure_coef_vuln", "vulnerability_coef"])
        if model_spec.get("exposure_split", False):
            exposure_coef_pop = pm.Normal("exposure_coef_pop", sigma=10)
            exposure_coef_wpc = pm.Normal("exposure_coef_wpc", sigma=10)
            mu = mu + exposure_coef_pop*np.log(population/area) + exposure_coef_wpc*np.log(WPC)
            var_names.extend(["exposure_coef_pop", "exposure_coef_wpc"])
        # if model_spec.get("exposure_poly", False):
        #     exposure_coef_poly1 = pm.Normal("exposure_coef_poly1", sigma=10)
        #     exposure_coef_poly2 = pm.Normal("exposure_coef_poly2", sigma=10)
        #     mu = mu + exposure_coef_poly1*np.log(population*WPC/area) + exposure_coef_poly2*(np.log(population*WPC/area))**2
        #     var_names.extend(["exposure_coef_poly1", "exposure_coef_poly2"])
        if model_spec.get("pressure", False):
            # Linear in log-space: log(damage) ~ coef * pressure
            pressure_coef = pm.Normal("pressure_coef", sigma=10)
            mu = mu + pressure_coef*delta_P_relative
            var_names.append("pressure_coef")
        if model_spec.get("pressure_poly_2", False):
            # Polynomial in log-space: log(damage) ~ coef1 * pressure + coef2 * pressure^2
            pressure_poly_coef1 = pm.Normal("pressure_poly_coef1", sigma=10)
            pressure_poly_coef2 = pm.Normal("pressure_poly_coef2", sigma=10)
            mu = mu + pressure_poly_coef1*delta_P_relative + pressure_poly_coef2*delta_P_relative**2
            var_names.extend(["pressure_poly_coef1", "pressure_poly_coef2"])
        if model_spec.get("pressure_poly_3", False):
            # Cubic polynomial in log-space: log(damage) ~ coef1 * pressure + coef2 * pressure^2 + coef3 * pressure^3
            pressure_poly_coef1 = pm.Normal("pressure_poly_coef1", sigma=10)
            pressure_poly_coef2 = pm.Normal("pressure_poly_coef2", sigma=10)
            pressure_poly_coef3 = pm.Normal("pressure_poly_coef3", sigma=10)
            mu = mu + pressure_poly_coef1*delta_P_relative + pressure_poly_coef2*delta_P_relative**2 + pressure_poly_coef3*delta_P_relative**3
            var_names.extend(["pressure_poly_coef1", "pressure_poly_coef2", "pressure_poly_coef3"])
        
        if model_spec.get("pressure_poly_4", False):
            # Quartic polynomial in log-space: log(damage) ~ coef1 * pressure + coef2 * pressure^2 + coef3 * pressure^3 + coef4 * pressure^4
            pressure_poly_coef1 = pm.Normal("pressure_poly_coef1", sigma=10)
            pressure_poly_coef2 = pm.Normal("pressure_poly_coef2", sigma=10)
            pressure_poly_coef3 = pm.Normal("pressure_poly_coef3", sigma=10)
            pressure_poly_coef4 = pm.Normal("pressure_poly_coef4", sigma=10)
            mu = mu + pressure_poly_coef1*delta_P_relative + pressure_poly_coef2*delta_P_relative**2 + pressure_poly_coef3*delta_P_relative**3 + pressure_poly_coef4*delta_P_relative**4
            var_names.extend(["pressure_poly_coef1", "pressure_poly_coef2", "pressure_poly_coef3", "pressure_poly_coef4"])
        if model_spec.get("pressure_poly_trend", False):
            pressure_poly_coef1 = pm.Normal("pressure_poly_coef1", sigma=10)
            pressure_poly_coef2 = pm.Normal("pressure_poly_coef2", sigma=10)
            interaction_term = pm.Normal("i_term")
            mu = mu + interaction_term*(pressure_poly_coef1*delta_P_relative )*np.array(delta_years) + pressure_poly_coef2*delta_P_relative**2
            var_names.extend(["pressure_poly_coef1", "pressure_poly_coef2", "i_term"])
        if model_spec.get("pressure_threshold", False):
            # Threshold response in log-space: only log(pressure) above threshold
            pressure_thresh = pm.Normal("pressure_threshold_val", mu=2, sigma=2)
            pressure_thresh_coef = pm.Normal("pressure_thresh_coef", sigma=3)
            pressure_excess = pm.math.maximum(delta_P_relative - pressure_thresh, 0)
            mu = mu + pressure_thresh_coef * pressure_excess
            var_names.extend(["pressure_threshold_val", "pressure_thresh_coef"])


        
        if model_spec.get("tides", False):
            tides_coef = pm.Normal("tides_coef", sigma=10)
            mu = mu + tides_m*tides_coef
            var_names.append("tides_coef")
        
        if model_spec.get("wind", False):
            wind_coef = pm.Normal("wind_coef", sigma=10)
            mu = mu + wind_coef*wind_speed_relative
            var_names.append("wind_coef")


        if model_spec.get("residual_wind", False):
            wind_coef_residual = pm.Normal("wind_coef_residual", sigma=5)
            mu = mu + wind_coef_residual*residual_wind_relative
        if model_spec.get("wind_power_law", False):
            #prior here: chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.nature.com/articles/nature07234.pdf?utm_source=wiley&getft_integrator=wiley
            wind_power_law_coef = pm.Normal("wind_power_law_coef", mu=9, sigma=4)
            raw_wind_ms = wind_speed_raw * 0.514444 #convert knots to m/s
            wind_power_law = np.log(raw_wind_ms)*(wind_power_law_coef)
            mu = mu + wind_power_law
            var_names.append("wind_power_law_coef")
        if model_spec.get("wind_expert_exp", False):
            #prior here : chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/https://www.nature.com/articles/nature07234.pdf?utm_source=wiley&getft_integrator=wiley
            wind_expert_exp_coef = pm.Normal("wind_expert_exp_coef", mu=0.05, sigma=0.1)
            raw_wind_ms = wind_speed_raw * 0.514444 #convert knots to m/s
            mu = mu + wind_expert_exp_coef *raw_wind_ms # Add small constant to avoid log(0)
            var_names.append("wind_expert_exp_coef")
        
        if model_spec.get("travel_speed", False):
            travel_speed_coef = pm.Normal("travel_speed_coef", sigma=10)
            mu = mu + travel_speed_coef*travel_speed
            var_names.append("travel_speed_coef")

        if model_spec.get("trend", False):
            trend_coef = pm.Normal("trend_coef", mu=0, sigma=0.01)
            mu = mu + trend_coef*np.array(delta_years)
            var_names.append("trend_coef")
        if model_spec.get("trend_centered", False):
            trend_centered_coef = pm.Normal("trend_centered_coef", mu=0, sigma=0.01)
            years_centered = np.array(delta_years) - np.mean(delta_years)
            mu = mu + trend_centered_coef*years_centered
            var_names.append("trend_centered_coef")
        if model_spec.get("c_trend_pressure_interaction", False):
            beta_1 = pm.Normal("beta_1", sigma=3)
            beta_2 = pm.Normal("beta_2", sigma=3)
            beta_3 = pm.Normal("beta_3", sigma=3)
            years_centered = np.array(delta_years) - np.mean(delta_years)
            delta_P_relative_centered = delta_P_relative - np.mean(delta_P_relative)
            mu = mu + beta_1 * delta_P_relative_centered * years_centered + beta_2 * delta_P_relative_centered + beta_3 * years_centered
            var_names.extend(["beta_1", "beta_2", "beta_3"])
        if model_spec.get("pressure_with_trend", False):
            pressure_coef = pm.Normal("pressure_coef", sigma=3)
            pressure_trend_interaction_coef = pm.Normal("pressure_trend_interaction_coef", sigma=0.1)
            mu = mu + pressure_coef * delta_P_relative + pressure_trend_interaction_coef * delta_P_relative * np.array(delta_years)
            var_names.extend(["pressure_trend_interaction_coef", "pressure_coef"])
        if model_spec.get("exposure_with_trend", False):
            exposure_trend_interaction_coef = pm.Normal("exposure_trend_interaction_coef", sigma=10)
            exposure_coeff = pm.Normal("exposure_coef", sigma=10)
            mu = mu + exposure_coeff*np.log(population*WPC/area) + exposure_trend_interaction_coef * np.log(population*WPC/area) * np.array(delta_years)
            var_names.extend(["exposure_trend_interaction_coef", "exposure_coef"])
        if model_spec.get("inverse_barometer", False):
            ib_coef = pm.Normal("ib_coef", sigma=10)
            mu = mu + ib_coef*ib_vals_relative  # Example term for inverse barometer effect
            var_names.append("ib_coef")
        if model_spec.get("exposure_raw", False):
            mu = mu + np.log(population*WPC/area)

        if model_spec.get("temp_anomaly_global", False):
            temp_anomaly_global_coef = pm.Normal("temp_anomaly_global_coef", sigma=10)
            mu = mu + temp_anomaly_global_coef*temp_anomaly_global
            var_names.append("temp_anomaly_global_coef")
        if model_spec.get("seasonal", False):
            season_sin_coef = pm.Normal("season_sin_coef", sigma=10)
            season_cos_coef = pm.Normal("season_cos_coef", sigma=10)
            mu = mu + season_sin_coef * season_sin
            mu = mu + season_cos_coef * season_cos 
            var_names.extend(["season_sin_coef", "season_cos_coef"])
        if model_spec.get("pressure_trend_interaction", False):
            pressure_trend_interaction_coef = pm.HalfNormal("pressure_trend_interaction_coef", sigma=10)
            mu = mu + pressure_trend_interaction_coef * delta_P_relative * np.array(delta_years)
            var_names.append("pressure_trend_interaction_coef")
        if model_spec.get("gc_hadisst", False):
            temp_coef = pm.Normal("coef_gc_hadisst", sigma=10)
            mu = mu + temp_coef * temp_hadisst_gc
            var_names.append("coef_gc_hadisst")
        if model_spec.get("mdr_hadisst", False):
            temp_coef_mdr = pm.Normal("coef_mdr_hadisst", sigma=10)
            mu = mu + temp_coef_mdr * temp_hadisst_mdr
            var_names.append("coef_mdr_hadisst")
        if model_spec.get("gc_icoads", False):
            temp_coef = pm.Normal("coef_gc_icoads", sigma=10)
            mu = mu + temp_coef * temp_icoads_gc
            var_names.append("coef_gc_icoads")
        if model_spec.get("mdr_icoads", False):
            temp_coef_mdr = pm.Normal("coef_mdr_icoads", sigma=10)
            mu = mu + temp_coef_mdr * temp_icoads_mdr
            var_names.append("coef_mdr_icoads")
        if model_spec.get("sea_level_rise", False):
            slope_sea_level = 0.0025  # meters per year
            sea_level_rise = slope_sea_level * np.array(delta_years)
            sea_level_coef = pm.Normal("sea_level_coef", sigma=10)
            mu = mu + sea_level_coef * sea_level_rise
            
        if model_spec.get("ibtracks_speed", False):
            ibtracks_speed_coef = pm.Normal("ibtracks_speed_coef", sigma=10)
            ibtracks_speed = df['STORM_SPEED_ms'].values
            mu = mu + ibtracks_speed_coef * (ibtracks_speed / 10)  # normalize to 10 m/s
            var_names.append("ibtracks_speed_coef")

        if model_spec.get("wind_vulnerability", False):
            #prior here: https://journals.ametsoc.org/view/journals/wcas/3/4/wcas-d-11-00007_1.xml
            v_threshold = pm.Normal("v_threshold", mu=50, sigma=10)
            v_half = pm.Normal("v_half", mu=120, sigma=15)
            v_coef = pm.Normal("v_coef", sigma=10)
            v_n = pm.math.maximum(wind_speed_raw-v_threshold,0)/(v_half-v_threshold)
            vulnerability = v_n**3/(1+v_n**3)
            mu = mu + v_coef*np.log(area*vulnerability + 1e-6)  # add small constant to avoid log(0)
            var_names.extend(["v_threshold", "v_half", "v_coef"])
        if model_spec.get('r34_mean', False):
            r34_coef = pm.Normal("r34_coef", sigma=10)
            mu = mu + r34_coef * r34_mean
            var_names.append("r34_coef")
        if model_spec.get('r50_mean', False):
            r50_coef = pm.Normal("r50_coef", sigma=10)
            mu = mu + r50_coef * r50_mean
            var_names.append("r50_coef")
        if model_spec.get('r64_mean', False):
            r64_coef = pm.Normal("r64_coef", sigma=10)
            mu = mu + r64_coef * r64_mean
            var_names.append("r64_coef")
        if model_spec.get("vulnerability", False):
            # Calculate damage normalized by exposed value in each state
            # damage_to_exposure = basedamage / (population * WPC)
            # This removes exposure growth confounding
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
            var_names.append("state_vuln_coef")
        if model_spec.get("risk_score", False):
            risk_score_coef = pm.Normal("risk_score_coef", sigma=10)
            mu = mu + risk_score_coef * risk_scores
            var_names.append("risk_score_coef")

        if model_spec.get("fitted_exposure", False):
            fitted_exposure_coef = pm.Normal("fitted_exposure_coef", sigma=10)
            fitted_exposure = df_clean['fitted_exposure'].values
            mu = mu + fitted_exposure_coef * fitted_exposure
            var_names.append("fitted_exposure_coef")
        #maybe remove these:
        if model_spec.get("modelled_wind", False):

            # Use the inferred pressure_true to predict wind
            modelled_wind_raw = equation_11_wind(pressure_raw, df_clean['lf_lat'].values, travel_speed)
            modelled_wind_relative = modelled_wind_raw / category_1_wind_baseline

            residual_wind_raw = wind_speed_raw - modelled_wind_raw
            residual_wind_relative = residual_wind_raw / category_1_wind_baseline
            
            # Add both to your model
            modelled_wind_coef = pm.Normal("modelled_wind_coef", sigma=5)
            wind_coef_residual = pm.Normal("wind_coef_residual", sigma=5)
            mu = mu + modelled_wind_coef * modelled_wind_relative
            mu = mu + wind_coef_residual * residual_wind_relative
        
        if model_spec.get("state_effects", False):

            states = df_clean['lf_state'].values
            unique_states, state_idx = np.unique(states, return_inverse=True)
            n_states = len(unique_states)

            sigma_state = pm.Exponential("sigma_state", 1.0)

            state_raw = pm.Normal("state_raw", 0, 1, shape=n_states)

            state_effect = pm.Deterministic(
                "state_effect",
                state_raw * sigma_state
            )
            

            mu = mu + state_effect[state_idx]

            var_names.extend(["sigma_state", "state_effect"])
        if model_spec.get("pressure_bart", False):
            linear_coef = pm.Normal("linear_coef", sigma=10)
            saturation_point = pm.Normal("saturation_point", sigma=2)  # Informative prior
            
            below_threshold = linear_coef * delta_P_relative
            at_saturation = linear_coef * saturation_point
            saturated_response = pytensor.tensor.switch(
                delta_P_relative < saturation_point,
                below_threshold,
                at_saturation
            )
            mu = mu + saturated_response
            var_names.extend(["saturation_point", "linear_coef"])
        if model_spec.get("surge_full_bart", False):
            surge_mu = df_clean['Surge_m_full_conversion_AN_as_surge'].values
            surge_uncertainty = df_clean['Uncertainty_m_full_conversion_AN_as_surge'].values
            surge_true = pm.TruncatedNormal(
                "surge_true",
                mu=surge_mu,
                sigma=surge_uncertainty,
                lower=0.0,
                shape=len(surge_mu)
            )


            surge_coef = pm.Normal("surge_coef", sigma=10)
            surge_effect = surge_coef * surge_true
            saturation_point = pm.Normal("saturation_point_surge", mu=5.5, sigma=0.5)  # Informative prior
            
            below_threshold = surge_effect
            at_saturation = surge_coef * saturation_point
            saturated_response = pytensor.tensor.switch(
                surge_effect < saturation_point,
                below_threshold,
                at_saturation
            )
            mu = mu + saturated_response
            var_names.extend(["saturation_point_surge", "surge_coef"])
        if model_spec.get("pressure_bart2", False):
                # Early stage: polynomial (curved response at low pressure)
                pressure_poly_coef1 = pm.Normal("pressure_poly_coef1", sigma=10)
                pressure_poly_coef2 = pm.Normal("pressure_poly_coef2", sigma=10)
                
                # Transition point
                transition_point = pm.Normal("transition_point", mu=0.7, sigma=0.1)
                
                # Linear term (takes over at higher pressures)
                pressure_linear_coef = pm.Normal("pressure_linear_coef", mu=2, sigma=2)
                
                # Saturation point
                saturation_point = pm.Normal("saturation_point", mu=2.1, sigma=0.3)
                
                # Piecewise logic
                poly_part = pressure_poly_coef1 * delta_P_relative + pressure_poly_coef2 * (delta_P_relative**2)
                linear_part = pressure_linear_coef * delta_P_relative
                saturated_part = pressure_linear_coef * saturation_point  # Effect caps at saturation point
                
                # Use switch statements to piece together
                pressure_effect = pytensor.tensor.switch(
                    delta_P_relative < transition_point,
                    poly_part,  # polynomial regime
                    pytensor.tensor.switch(
                        delta_P_relative < saturation_point,
                        linear_part,  # linear regime
                        saturated_part  # saturation regime
                    )
                )
                mu = mu + pressure_effect
                var_names.extend(["pressure_poly_coef1", "pressure_poly_coef2", 
                                "transition_point", "pressure_linear_coef",
                                "saturation_point",])
        if model_spec.get("pressure_bart3", False):
            # Smooth transition from LINEAR → SATURATED by sigmoid
            pressure_linear_coef = pm.Normal("pressure_linear_coef", sigma=5)
            saturation_level = pm.Normal("saturation_level_pressure", mu=2.5, sigma=0.5)
            transition_steepness = pm.HalfNormal("transition_steepness", sigma=2)
            transition_point = pm.Normal("transition_point", mu=1.5, sigma=0.5)
            linear_response = pressure_linear_coef * delta_P_relative
            sigmoid_transition = 1 / (1 + pm.math.exp(-transition_steepness * (delta_P_relative - transition_point)))
            pressure_effect = linear_response * (1 - sigmoid_transition) + saturation_level * sigmoid_transition
            mu = mu + pressure_effect
            var_names.extend(["pressure_linear_coef", "saturation_level_pressure", "transition_steepness", "transition_point"])
        if model_spec.get("pressure_bart3_trend", False):
            # Smooth transition from LINEAR → SATURATED by sigmoid, with trend interaction
            pressure_linear_coef = pm.Normal("pressure_linear_coef", sigma=5)
            saturation_level = pm.Normal("saturation_level_pressure", mu=2.5, sigma=0.5)
            transition_steepness = pm.HalfNormal("transition_steepness", sigma=2)
            transition_point = pm.Normal("transition_point", mu=1.5, sigma=0.5)
            trend_interaction_coef = pm.Normal("trend_interaction_coef", sigma=0.1)
            linear_response = pressure_linear_coef * delta_P_relative
            sigmoid_transition = 1 / (1 + pm.math.exp(-transition_steepness * (delta_P_relative - transition_point)))
            pressure_effect = linear_response * (1 - sigmoid_transition) + saturation_level * sigmoid_transition
            interaction_effect = trend_interaction_coef * delta_P_relative * np.array(delta_years) * (1 - sigmoid_transition)
            mu = mu + pressure_effect + interaction_effect
            var_names.extend(["pressure_linear_coef", "saturation_level_pressure", "transition_steepness", "transition_point", "trend_interaction_coef"])
        if model_spec.get("exposure_bart", False):
            # Exposure flat until log(exposure)=10, then increases linearly
            exposure_log = np.log(population*WPC/area)
            
            # Hard threshold at log(exposure)=10
            exposure_threshold = pm.Normal("exposure_threshold", mu=10, sigma=2)
            exposure_coef = pm.Normal("exposure_coef", sigma=2)
            
            # Effect only grows above threshold
            exposure_excess = pm.math.maximum(exposure_log - exposure_threshold, 0)
            exposure_effect = exposure_log + exposure_coef * exposure_excess
            
            mu = mu + exposure_effect
            var_names.extend(["exposure_threshold", "exposure_coef", ])
        if model_spec.get("exposure_bart2", False):
            # Exposure flat until log(exposure)=10, then increases linearly
            exposure_log = np.log(population*WPC/area)
            
            # Hard threshold at log(exposure)=10
            exposure_threshold = pm.Normal("exposure_threshold", mu=10, sigma=2)
            exposure_coef1 = pm.Normal("exposure_coef1", sigma=2)
            exposure_coef2 = pm.Normal("exposure_coef2", sigma=2)
            pre_threshold_effect = exposure_coef1 * exposure_log
            post_threshold_effect = exposure_coef2 * exposure_log
            exposure_effect = pm.math.switch(exposure_log < exposure_threshold, pre_threshold_effect, post_threshold_effect)
            mu = mu + exposure_effect
            var_names.extend(["exposure_threshold", "exposure_coef1", "exposure_coef2"])

        # --- SELECT LIKELIHOOD BASED ON model_spec ---
        if likelihood == "log_studentt":
            # Log-Student's t likelihood (handles heavy tails / outliers)
            #nu = pm.Exponential("nu", lam=0.1)  # Mean of 10, allowing flexibility
            nu = pm.HalfNormal("nu", sigma=15)  # Alternative: HalfNormal for more mass at higher values
            
            obs = pm.CustomDist("obs",
                                 mu, sigma_total, nu,
                                 logp=log_studentt_logp,
                                 random=log_studentt_random,
                                 observed=np.exp(observed))
            var_names.append("nu")
            
        elif likelihood == "ulln":
            # Upper Limit Log-Normal likelihood
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
            var_names.append("upper")
        if likelihood == "pareto":
            # Pareto likelihood (heavy-tailed, only positive values)
            scale = pm.HalfNormal("scale", sigma=10)
            alpha = pm.Exponential("a", lam=1.0)
            obs = pm.Pareto("obs",
                                  alpha, scale,
                                 observed=np.exp(observed))
            var_names.extend(["scale", "a"])
        elif likelihood == "lognormal":
            # Standard log-normal likelihood (default)
            obs = pm.LogNormal("obs", mu=mu, sigma=sigma_total, observed=np.exp(observed))
        
        # --- Prior predictive check: draw from priors before sampling ---

        # prior = pm.sample_prior_predictive(samples=10000, random_seed=42)

        # # extract prior predictive draws for observed outcome 'obs'
        # prior_obs = None
        # if isinstance(prior, dict):
        #     prior_obs = prior.get("obs", None)
        # else:
        #     try:
        #         prior_obs = prior.prior_predictive["obs"].values
        #     except Exception:
        #         prior_obs = None

        # if prior_obs is not None:
        #     prior_vals = prior_obs.flatten()
        #     # Safely convert to log-space. Only log-transform if a majority of draws are positive.
        #     prior_vals = prior_vals[np.isfinite(prior_vals)]
        #     n = len(prior_vals)
        #     n_pos = int((prior_vals > 0).sum())
        #     if n > 0 and n_pos >= max(1, int(0.5 * n)):
        #         # log-transform positive draws; drop non-positive values
        #         prior_vals_plot = np.log(prior_vals[prior_vals > 0] + 1e-12)
        #     else:
        #         # fallback: assume prior draws are already on the log scale
        #         prior_vals_plot = prior_vals
        #     if prior_vals_plot.size == 0:
        #         prior_vals_plot = np.array([np.nan])
        #     combined = np.concatenate([observed, prior_vals_plot])
        #     bins = np.linspace(np.nanmin(combined), np.nanmax(combined), max(int(np.sqrt(len(observed))), 25))
        #     plt.figure(figsize=(10,6))
        #     plt.hist(prior_vals_plot, bins=bins, density=True, alpha=0.5, label="Prior predictive (log or fallback)")
        #     plt.hist(observed, bins=bins, density=True, alpha=0.5, label="Observed (log)")
        #     plt.xlabel("ln(Base Damage)")
        #     plt.ylabel("Density")
        #     plt.title(f"Prior Predictive vs Observed ({model_name})")
        #     plt.legend()
        #     plt.savefig(os.path.join(model_path, f"{filename}_prior_predictive_hist.png"))
        #     plt.close()

        # # Optional: prior predictive for surge_true if present
        # prior_surge = None
        # if isinstance(prior, dict):
        #     prior_surge = prior.get("surge_true", None)
        # else:
        #     try:
        #         prior_surge = prior.prior_predictive["surge_true"].values
        #     except Exception:
        #         prior_surge = None

        # if prior_surge is not None:
        #     plt.figure(figsize=(6,4))
        #     plt.hist(prior_surge.flatten(), bins=50, density=True, alpha=0.7)
        #     plt.title("Prior predictive: surge_true")
        #     plt.savefig(os.path.join(model_path, f"{filename}_prior_surge_true_hist.png"))
        #     plt.close()

        trace = pm.sample(draws=1500, tune=1500, target_accept=0.95, idata_kwargs={'log_likelihood':True}, compile_kwargs={"mode": "NUMBA"})
        summary = az.summary(
            trace,
            hdi_prob=0.95,
        )
        summary.to_csv(os.path.join(model_path, f"{filename}_summary.csv"))
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(summary)
    

    
    axes = az.plot_trace(trace, var_names=var_names, figsize=(12, 12))
    fig = np.asarray(axes).ravel()[0].get_figure()

    fig.subplots_adjust(hspace=0.8, wspace=0.25)  # <- increase hspace for more vertical room
    fig.savefig(os.path.join(model_path, f"{filename}_trace.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    
    az.plot_pair(trace, var_names=var_names, kind="kde", marginals=True, divergences=True)
    plt.savefig(os.path.join(model_path, f"{filename}_pair.png"))
    plt.close()
    
    # # Posterior predictive check
    # with model:
    #     ppc = pm.sample_posterior_predictive(trace)
    
    # ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    # # Histogram comparison
    # if use_ulln:
    #     # ppc_values are in original scale, observed is in log scale
    #     ppc_values_log = np.log(ppc_values)
    #     combined = np.concatenate([observed, ppc_values_log])
    #     bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(observed))), 25))
    #     plt.figure(figsize=(10,6))
    #     plt.hist(observed, bins=bins, density=True, alpha=0.5, label="Observed")
    #     plt.hist(ppc_values_log, bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    # else:
    #     # Both are already in log scale
    #     combined = np.concatenate([observed, ppc_values])
    #     bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(observed))), 25))
    #     plt.figure(figsize=(10,6))
    #     plt.hist(observed, bins=bins, density=True, alpha=0.5, label="Observed")
    #     plt.hist(ppc_values, bins=bins, density=True, alpha=0.5, label="Posterior predictive")

    # plt.xlabel("ln(Base Damage)")
    # plt.ylabel("Density")
    # plt.title(f"Posterior Predictive Histogram ({model_name})")
    # plt.legend()
    # plt.savefig(os.path.join(model_path, f"{filename}_ppc_histogram.png"))
    # plt.close()
    
    return trace, model_name



def bart_model(df, observed_variable='basedamage', m=175):
    """
    Fit a BART model using PyMC.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    observed_variable : str
        Name of observed variable (default 'basedamage')
    m : int
        Number of trees in BART ensemble (default 15)
    
    Returns:
    --------
    trace : InferenceData
        MCMC trace
    model_name : str
        Model identifier
    """
    # Define priors for the BART model
    df_clean = df.dropna(subset=[observed_variable, 'population', 'WPC', 'lf_pressure', 'lf_ISO_TIME', 'Surge_m_full_conversion_AN_as_surge']).copy()
    df_clean = df_clean[df_clean[observed_variable] > 0].copy()

    # Target
    y = np.log(df_clean[observed_variable].values)
    # Features (keep similar to your parametric model!)
    timestamps = df_clean['lf_ISO_TIME'].values 
    years = pd.to_datetime(timestamps).year
    delta_years = [int(year - int(years[0])) for year in years]
    category_1_pressure_baseline = 980
    pressure_raw = df_clean['lf_pressure'].values
    delta_P = (1013.25-pressure_raw)* 100  #convert to Pa
    delta_P_baseline = (1013.25 - category_1_pressure_baseline)*100
    delta_P_relative = delta_P / delta_P_baseline  #Pa

    exposure = np.log(df_clean['population'].values * df_clean['WPC'].values / 10000)

    surge = df_clean['Surge_m_full_conversion_AN_as_surge'].values
    X = pd.DataFrame({
        "delta_P_relative": delta_P_relative,
        "delta_years": delta_years,
        "exposure": exposure,
        "surge": surge
    })

    with pm.Model() as bart_model:
        mu_bart = BART("mu", X=X, m=m, Y=y)
        sigma = pm.HalfNormal("sigma", sigma=2)
        obs = pm.Normal("obs", mu=mu_bart, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=3000,
            tune=1000,
            chains=4,
            target_accept=0.9,
            idata_kwargs={"log_likelihood": True},
            random_seed=42,  # control parallelism
        )
        # Method 1: Using pymc_bart's built-in function (requires the model and trace)
    print("\n" + "="*60)
    print("COMPUTING VARIABLE IMPORTANCE")
    print("="*60)
    
    # Compute variable importance
    # Note: You need to pass the model and the variable name
    vi_results = compute_variable_importance(
        trace, 
        mu_bart,  # The BART variable name
        X
    )
    
    # Plot variable importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = plot_variable_importance(
        vi_results, 
        ax=ax,
        plot_kwargs={"rotation": 45},
        figsize=(10, 4)
    )
    ax.set_ylim(0, 1)  # Better range for importance
    ax.set_title(f'Variable Importance (m={m} trees)')
    plt.tight_layout()
    plt.savefig(f'bart_variable_importance_m{m}.png', dpi=150)
    plt.show()
    
    
    # ===== OPTIONAL: Partial Dependence Plots =====
    print("\n" + "="*60)
    print("GENERATING PARTIAL DEPENDENCE PLOTS")
    print("="*60)
    
    axes = plot_pdp(mu_bart, X=X, Y=y, grid=(2, 2), xs_interval="insample", figsize=(12, 5))
    plt.show()
    return trace, f"bart_m{m}"


def bart_loo_diagnostics(df, m_values=[10, 15, 20], observed_variable='basedamage'):
    """
    Compare BART models with different tree counts using LOO.
    
    Helps determine if m is sufficient or if you're underfitting/overfitting.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    m_values : list
        List of m (tree count) values to test
    observed_variable : str
        Name of observed variable
    
    Returns:
    --------
    loo_comparison_df : DataFrame
        LOO scores for each m value, sorted by elpd_loo
    """
    print("\n" + "="*70)
    print("BART LOO DIAGNOSTICS: Testing different tree counts")
    print("="*70)
    
    loo_results = []
    
    for m in m_values:
        print(f"\nFitting BART with m={m} trees...")
        trace, model_name = bart_model(df, observed_variable=observed_variable, m=m)
        
        # Compute LOO
        loo = az.loo(trace)
        
        loo_results.append({
            "m_trees": m,
            "model_name": model_name,
            "elpd_loo": loo.elpd_loo,
        })
        print(az.summary(trace, var_names=["sigma","mu"]))
        path = str(m) + '_run.csv'
        az.summary(trace, var_names=["sigma","mu"]).to_csv(path)

        
        print(f"  elpd_loo: {loo.elpd_loo:.2f} ")
    
    loo_comparison_df = pd.DataFrame(loo_results).sort_values("elpd_loo", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*70)
    print("BART LOO COMPARISON SUMMARY")
    print("="*70)
    print(loo_comparison_df)
    print("\nInterpretation:")
    print("  - If LOO improves monotonically, increase m further")
    print("  - If LOO plateaus, use the smallest m (parsimony)")
    print("  - If BART LOO >> best parametric model, you're missing interactions")
    print("="*70 + "\n")
    
    return loo_comparison_df


def compare_models(df, model_specs, observed_variable='basedamage'):
    """
    Compare multiple model specifications using LOO-CV.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    model_specs : list of dicts
        List of model specification dicts. Each dict should contain:
        - Feature flags: {"exposure": True/False, "pressure": True/False, ...}
        - Optional "likelihood" key: "lognormal" (default), "log_studentt", or "ulln"
        
        Example:
            [
                {"exposure": True, "pressure": True, "likelihood": "lognormal"},
                {"exposure": True, "pressure": True, "likelihood": "log_studentt"},
                {"exposure": True, "wind": True, "likelihood": "ulln"},
            ]
    
    observed_variable : str, optional
        Which variable to use as observation (default: 'basedamage')
    
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
    #not all hurricane has observed surge
    print(df.shape)
    surge_flag = True
    for model_spec in model_specs:
        if model_spec.get("surge_full", False):
            surge_flag = True
            break
        if model_spec.get("latent_intensity", False):
            surge_flag = True
            break
    if surge_flag:
        df = df.dropna(subset=['Surge_m_full_conversion_AN_as_surge', 'Uncertainty_m_full_conversion_AN_as_surge'])

    
    for i, spec in enumerate(model_specs, 1):
        print(f"\n[{i}/{len(model_specs)}] Fitting model with spec: {spec}")
        trace, model_name = hurricane_physical_model(df, model_spec=spec, observed_variable=observed_variable)
        traces[model_name] = trace

    # Compare models using WAIC
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS (LOO)")
    print("="*80)
    comparison_df = az.compare(traces, ic="loo", method="stacking", var_name="obs")  # Use LOO for comparison
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


def compare_model_by_period(df, model_spec, cutoff_year):
    """
    Fit same model to before/after periods and compare sigma, coefficients, fit quality.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    model_spec : dict
        Model specification including feature flags and optional "likelihood" key
    cutoff_year : int
        Year to split the data
    
    Returns:
    --------
    traces : dict
        Dictionary with "before" and "after" traces
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
        trace, model_name = hurricane_physical_model(df_period, model_spec=model_spec)
        traces[period_name] = trace

    
    return traces

    

from scipy.stats import norm
import numpy as np

def forward_search_with_bias_correction(df, model_specs_nested, observed_variable='basedamage', alpha=0.5, bias_correction_factor=1.5):
    """
    Forward search using LOO-based ELPD with selection-induced bias correction.
    Implements Sivula et al. 2025 Equations (17) and (18).
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    model_specs_nested : list
        List of nested model specs (increasing complexity)
    observed_variable : str
        Outcome variable name
    alpha : float
        Safety parameter (0.5 = maximum safety)
    bias_correction_factor : float
        Multiplier for bias correction (default 1.5 from Watanabe 2010, can test {1, 1.5, 2})
    
    Returns:
    --------
    results_df : DataFrame
        Results with uncorrected and corrected ELPD values
    saturation_idx : int or None
        Index where saturation occurs (first model with detectable bias)
    """
    K = len(model_specs_nested)
    search_results = []
    all_elpd_diffs = []  # Store ALL incremental differences for later correction
    
    print("\n" + "="*80)
    print(f"FORWARD SEARCH: {K} Nested Models with LOO + Bias Correction")
    print(f"(Sivula et al. 2025, Equations 17-18, bias_factor={bias_correction_factor})")
    print("="*80)
    
    # ===== PHASE 1: Fit all models and collect incremental differences =====
    for model_idx, model_spec in enumerate(model_specs_nested):
        print(f"\n[Step {model_idx+1}/{K}] Fitting model...")
        
        # Fit model once
        trace, model_name = hurricane_physical_model(
            df, 
            model_spec=model_spec, 
            observed_variable=observed_variable
        )
        
        # Compute LOO with pointwise values for bias correction
        loo_result = az.loo(trace, pointwise=True)
        mean_elpd = loo_result.elpd_loo
        se_elpd = loo_result.se
        pointwise_elpd = loo_result.loo_i.values  # Shape: (n_data_points,)
        
        result = {
            'model_index': model_idx + 1,
            'model_name': model_name,
            'mean_elpd_loo': mean_elpd,
            'se_elpd': se_elpd,
        }
        
        # Incremental if not first model
        if model_idx > 0:
            prev_pointwise = search_results[model_idx-1]['pointwise_elpd']
            
            # POINTWISE incremental ELPD (Sivula et al. Eq. 18)
            Delta_elpd_pointwise = pointwise_elpd - prev_pointwise
            Delta_elpd_loo = np.sum(Delta_elpd_pointwise)
            
            # SE from pointwise differences (Sivula et al.)
            n = len(Delta_elpd_pointwise)
            mean_delta = Delta_elpd_loo / n
            var_delta = np.sum((Delta_elpd_pointwise - mean_delta)**2) / (n - 1)
            se_delta = np.sqrt(var_delta * n)
            
            result['delta_elpd_loo'] = Delta_elpd_loo
            result['se_delta'] = se_delta
            
            all_elpd_diffs.append(Delta_elpd_loo)
        else:
            result['delta_elpd_loo'] = 0.0
            result['se_delta'] = np.inf
            result['cumulative_elpd_corrected'] = mean_elpd
        
        # Store pointwise for next iteration
        result['pointwise_elpd'] = pointwise_elpd
        
        search_results.append(result)
        
        print(f"  ELPD: {mean_elpd:.2f} ± {se_elpd:.2f}")
        if model_idx > 0:
            print(f"  Incremental: {Delta_elpd_loo:.2f} ± {se_delta:.2f}")
    
    # ===== PHASE 2: Apply per-step bias correction (Equations 17-18) =====
    print("\n" + "="*80)
    print("SELECTION-INDUCED BIAS CORRECTION (Sivula et al. 2025, Eq. 17-18)")
    print("="*80)
    
    if len(all_elpd_diffs) > 0:
        all_elpd_diffs = np.array(all_elpd_diffs)
        
        # Base ELPD at step 0
        elpd_cumulative_corrected = search_results[0]['mean_elpd_loo']
        
        print("\n" + "-"*110)
        print(f"{'Step':<6} {'Model':<30} {'ΔElpd_LOO':<13} {'S(k)σ̂_k':<12} {'Bias_mag':<12} {'Applied?':<10} {'ΔCorr':<13} {'Cumulative':<13}")
        print("-"*110)
        
        saturation_idx = None
        
        # Print step 0
        model_name_0 = search_results[0]['model_name'][:28]
        print(f"{'0':<6} {model_name_0:<30} {'—':<13} {'—':<12} {'—':<12} {'—':<10} {'—':<13} {elpd_cumulative_corrected:>11.4f}")
        
        for step_k in range(1, len(search_results)):
            k_num_models = step_k + 1  # At step k, we have k+1 total models (0, 1, ..., k)
            
            # Compute S(k) and σ̂_k PER-STEP (Equations 14, 16)
            s_k, sigma_k, equiv_threshold = compute_selection_bias_correction(
                all_elpd_diffs[:step_k],  # Only differences UP TO this step
                K=k_num_models, 
                alpha=alpha
            )
            
            delta_elpd_loo = search_results[step_k]['delta_elpd_loo']
            se_delta = search_results[step_k]['se_delta']
            
            # Equation 17: Apply correction if small relative to noise threshold
            bias_magnitude = bias_correction_factor * s_k * sigma_k
            is_small = abs(delta_elpd_loo) < equiv_threshold
            
            if is_small:
                # Apply bias correction (subtract bias estimate)
                delta_elpd_corrected = delta_elpd_loo - bias_magnitude
                corrected_flag = "✓ YES"
            else:
                # No correction needed
                delta_elpd_corrected = delta_elpd_loo
                corrected_flag = "✗ NO"
            
            # Equation 18: Cumulative corrected ELPD
            elpd_cumulative_corrected += delta_elpd_corrected
            
            # Store results
            search_results[step_k]['s_k'] = s_k
            search_results[step_k]['sigma_k'] = sigma_k
            search_results[step_k]['equiv_threshold'] = equiv_threshold
            search_results[step_k]['bias_magnitude'] = bias_magnitude
            search_results[step_k]['delta_elpd_corrected'] = delta_elpd_corrected
            search_results[step_k]['cumulative_elpd_corrected'] = elpd_cumulative_corrected
            search_results[step_k]['correction_applied'] = is_small
            
            model_name_k = search_results[step_k]['model_name'][:28]
            print(f"{step_k:<6} {model_name_k:<30} {delta_elpd_loo:>11.4f}  {equiv_threshold:>10.4f}  {bias_magnitude:>10.4f}  {corrected_flag:<10} {delta_elpd_corrected:>11.4f}  {elpd_cumulative_corrected:>11.4f}")
            
            # Saturation: first step where bias correction was applied (small difference detected)
            if is_small and saturation_idx is None:
                saturation_idx = step_k
        
        print("-"*110)
        
        if saturation_idx is not None:
            print(f"\n⚠️  SATURATION DETECTED at Step {saturation_idx}")
            print(f"    Selection-induced bias first detected at model {saturation_idx+1}")
            print(f"    Recommended best model: {search_results[saturation_idx-1]['model_name']}")
            print(f"    (Adding beyond shows signs of overfitting to selection process)")
        else:
            print("\n✓ No saturation. All incremental improvements exceed threshold.")
    
    # Save results
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'pointwise_elpd'}
        for r in search_results
    ])
    results_df.to_csv(r"./Speciale/Code/Simulations/forward_search_results.csv", index=False)
    print(f"\n✓ Results saved to: ./Speciale/Code/Simulations/forward_search_results.csv")
    
    return results_df, saturation_idx


def compute_selection_bias_correction(Delta_elpd_values, K, alpha=0.5):
    """
    Sivula et al. 2025 equations (14), (15), (16) for per-step bias correction.
    
    Parameters:
    -----------
    Delta_elpd_values : array
        Array of incremental ELPD differences accumulated so far (up to current step)
    K : int
        Total number of candidate models AT THIS STEP
    alpha : float
        Safety parameter (0.5 = most conservative)
    
    Returns:
    --------
    s_K, sigma_K, equiv_threshold : tuple of floats
        Critical quantile, upper tail SD, and equivalence threshold
    """
    if len(Delta_elpd_values) == 0:
        return 0.0, 0.0, 0.0
    
    # Equation (16): Estimate sigma_K from upper half-tail
    m_K = np.median(Delta_elpd_values)
    upper_half = Delta_elpd_values[Delta_elpd_values >= m_K]
    
    if len(upper_half) == 0:
        sigma_K = 0.0
    else:
        sigma_K = np.sqrt((2.0 / K) * np.sum((upper_half - m_K)**2))
    
    # Equation (14): Critical quantile (inverse normal CDF)
    quantile_arg = (K - alpha) / (K - 2*alpha + 1)
    # Ensure quantile_arg is in valid range [0, 1]
    quantile_arg = np.clip(quantile_arg, 1e-6, 1 - 1e-6)
    s_K = norm.ppf(quantile_arg)
    
    # Equation (15): Equivalence threshold
    equiv_threshold = s_K * sigma_K
    
    return s_K, sigma_K, equiv_threshold

if __name__ == "__main__":

    #Example usage
    #df = generate_csv_data()
    df = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\final_hurricane_data_aslak.csv")

    print(df.shape)
    df_clean = df.dropna(subset=['population', 'WPC', 'lf_wind', 'lf_pressure','Surge_m_full_conversion_AN_as_surge', 'Uncertainty_m_full_conversion_AN_as_surge'])# 'Lat_db', 'Lon_db'])
    # ========================================================================
    # IMPROVED PARAMETRIC MODELS (Based on BART discoveries)
    # ========================================================================
    # These specifications incorporate nonlinear pressure & temporal effects
    # discovered by BART analysis. Test each to improve model ELPD.
    
    model_specs = [
        {"exposure": True, "pressure": True, "trend": True, "surge_full": True, "alpha": True},
        {"exposure": True, "pressure": True, "trend": True, "surge_full": True, "alpha": True, "likelihood": "log_studentt"},
        {"exposure": True, "pressure": True, "trend": True, "surge_full": True, "alpha": True, "likelihood": "ulln"},
        {"exposure": True, "pressure": True, "trend": True, "surge_full": True, "alpha": True, "likelihood": "pareto"},]
    #     # {"exposure": True, "pressure": True, "trend": True, "surge_full": True},
    #     # {"exposure": True, "pressure": True, "trend": True, "surge_full": True, "alpha": True},
    #     # {"exposure": True, "pressure": True, "trend": True, "surge_full": True, "pressure_trend_interaction": True},
    #     # {"exposure": True, "trend": True, "surge_full": True, "pressure_bart_2": True, "alpha": True},
    #     # {"exposure": True, "trend": True, "surge_full": True, "pressure_bart_3": True, "alpha": True, "sigma_pressure": True},

        
    # ]
    comparison_df, traces = compare_models(df_clean, model_specs, observed_variable='basedamage')
    #hurricane_physical_model(df_clean, model_spec={"exposure": True, "surge_full": True, "pressure_trend_interaction": True, "alpha": True, "sigma_pressure": True}, use_ulln=False, observed_variable='basedamage')
    # nested_models = [
    # {"exposure": True},
    # {"exposure": True, "pressure": True},
    # {"alpha": True, "exposure": True, "pressure": True},
    # {"alpha": True, "exposure": True, "pressure": True, "trend": True},
    # {"alpha": True, "exposure": True, "pressure": True, "surge_full": True},
    # # {"exposure": True, "pressure_bart3": True, "trend": True, "surge_full": True},
    # # {"exposure": True, "pressure_bart3": True, "trend": True, "surge_full": True, "alpha": True},
    # ]

    # results_df, saturation_idx = forward_search_with_bias_correction(
    #     df_clean,
    #     nested_models,
    #     alpha=0.5
    # )