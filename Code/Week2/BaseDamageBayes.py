import pymc as pm
import numpy as np
import scipy.stats as stats
import arviz as az
from matplotlib import pyplot as plt
import os
import pandas as pd
from pytensor.printing import Print
import sys
from io import StringIO





def test_standard_normal(data):
    log_data = np.log(data)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=15, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=log_data)
        

        
        trace = pm.sample( draws=2500, tune=1000, target_accept=0.95)



    

    az.plot_trace(trace, var_names=["mu", "sigma"], figsize=(12, 12))
    plt.show()
    az.plot_pair(trace, var_names=["mu", "sigma"], kind="kde", marginals=True, divergences=True)
    plt.show()
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    # -----------------------
    # Method 1: Standard plt.hist
    # -----------------------
    combined = np.concatenate([np.log(data), ppc_values])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(data))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(log_data, bins=bins, density=True, alpha=0.5, label="Observed")
    plt.hist(ppc_values, bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(ATD)")
    plt.ylabel("Density")
    plt.title("Posterior Predictive Histogram (plt.hist)")
    plt.legend()
    plt.show()

def hurricane_physical_model(df):
    path = r"./Speciale/Code/Week2/Plots"

    # Clean data
    df_clean = df[df['basedamage'] > 0].copy()
    df_clean = df_clean.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    
    # Prepare variables
    log_BD = np.log(df_clean['basedamage'])
    ATD = df_clean['ATD'].values
    population = df_clean['population'].values
    WPC = df_clean['WPC'].values
    wind_speed = df_clean['lf_wind'].values
    pressure = df_clean['lf_pressure'].values
    areas = ATD/np.exp(log_BD)*population*WPC
    tides = df_clean['Tide_Level'].values
    tides = tides
    

    filename = "test" #_population"
    path = os.path.join(r"./Speciale/Code/Week2/Plots", filename)
    os.makedirs(path, exist_ok=True)

    # category_1_wind_baseline = 74*0.868976242 #convert mph to knots
    # category_1_pressure_baseline = 980 #mb
    # wind_speed = wind_speed / category_1_wind_baseline
    #pressure = pressure / category_1_pressure_baseline

    with pm.Model() as model:
        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=17, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=7)
        wind_speed_coef = pm.Normal("wind_speed_coef", mu=0, sigma=2)
        pressure_coef = pm.Normal("pressure_coef", mu=0, sigma=1) #Doesnt add much, very correlated with wind_speed_coef
        tides_coef = pm.HalfNormal("tides_coef",  sigma=0.5) #Doesnt add much
        economic_const = pm.Normal("economic_const", mu=1, sigma=0.5) #Is a must to have alpha be positive
        #weight = pm.Uniform("weight", lower=0, upper=1)
        
        mu = alpha + np.log(population*WPC/areas) + pressure*pressure_coef + tides*tides_coef + wind_speed*wind_speed_coef
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=log_BD)
        
        trace = pm.sample(draws=2500, tune=1000, target_accept=0.95)
        summary = az.summary(trace, hdi_prob=0.95)
        summary.to_csv(os.path.join(path, filename + ".csv"))
        print(summary)

    # Plot traces
    az.plot_trace(trace, var_names=["alpha", "sigma", "wind_speed_coef", "pressure_coef", "tides_coef" ], figsize=(12, 12))
    plt.savefig(os.path.join(path, filename + "trace.png"))
    
    az.plot_pair(trace, var_names=["alpha", "sigma", "wind_speed_coef", "pressure_coef", "tides_coef"], 
                 kind="kde", marginals=True, divergences=True)
    plt.savefig(os.path.join(path, filename + "pair.png"))
    #plt.show()
    
    # Posterior predictive check
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    
    ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    
    # Histogram comparison
    combined = np.concatenate([log_BD, ppc_values])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(log_BD))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(log_BD, bins=bins, density=True, alpha=0.5, label="Observed")
    plt.hist(ppc_values, bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title("Posterior Predictive Histogram")
    plt.legend()
    plt.savefig(os.path.join(path, filename + "_ppc_histogram.png"))
    
    return trace

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
    

    wind_ms = wind_speed * 0.514444 #convert knots to m/s
    with pm.Model() as model:
        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=7)
        pressure_coef = pm.Normal("pressure_coef", mu=0, sigma=1) #Doesnt add much, very correlated with wind_speed_coef
        tides_coef = pm.HalfNormal("tides_coef",  sigma=0.5) #Doesnt add much
        b = pm.TruncatedNormal("b", mu=40.0, sigma=15.0, lower=1e-3)
        a = pm.TruncatedNormal("a", mu=10**(-3.5), sigma=1, lower=1e-8, upper=1e-2)
        c = pm.TruncatedNormal("c", mu=10.0, sigma=2.0, lower=0.1)

        mu = alpha + np.log(WPC*population/areas) + np.log(((wind_ms/b)**c + a)*WPC*population) +tides*tides_coef #+ pressure*pressure_coef
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=np.log(BD))
        
        trace = pm.sample(draws=2500, tune=1000, target_accept=0.95)
        summary = az.summary(trace, hdi_prob=0.95)
        print(summary)

    # Plot traces
    az.plot_trace(trace, var_names=["alpha", "sigma", "pressure_coef", "tides_coef" ], figsize=(12, 12))
    plt.show()
    az.plot_pair(trace, var_names=["alpha", "sigma", "pressure_coef", "tides_coef"], 
                 kind="kde", marginals=True, divergences=True)
    plt.show()
    
    # Posterior predictive check
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    
    ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    log_BD = np.log(BD)
    # Histogram comparison
    combined = np.concatenate([log_BD, np.log(ppc_values)])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(log_BD))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(log_BD, bins=bins, density=True, alpha=0.5, label="Observed")
    plt.hist(np.log(ppc_values), bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title("Posterior Predictive Histogram")
    plt.legend()
    plt.show()
    
    return trace


if __name__ == "__main__":
    #Example usage
    #df = pd.read_excel('./Speciale/Hurricane_data/Aslak_data.xls', sheet_name='ATD of ICAT', engine='xlrd')
    df = pd.read_csv('./Speciale/Hurricane_data/Aslak_data_with_tide.csv')
    df_clean = df.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    BD = np.array(df_clean['basedamage'].values)
    BD = BD[BD > 0]  #remove non-positive values

    plt.hist(np.log(BD), bins=20, density=True, alpha=0.6, color='g')
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title("Histogram of Base Damage")
    plt.savefig("./Speciale/Code/Week2/Plots/BaseDamage_histogram.png")
    plt.close()

    #hurricane_physical_model_APLR(df)
    hurricane_physical_model_APLR(df)
