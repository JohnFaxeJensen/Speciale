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



#os.chdir(r"./Speciale/Code/Week2")
def test_standard_lognormal(data):
    scale = 1e6
    scaled_data = data / scale

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=np.mean(np.log(scaled_data)), sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        obs = pm.LogNormal("obs", mu=mu, sigma=sigma, observed=scaled_data)
        
        start = {
            "mu": np.mean(np.log(scaled_data)),
            "sigma_log__": np.log(np.std(np.log(scaled_data)))
        }
        
        trace = pm.sample(start=start, draws=2500, tune=1000, target_accept=0.95)



    

    az.plot_trace(trace, var_names=["mu", "sigma"], figsize=(12, 12))
    plt.show()
    az.plot_pair(trace, var_names=["mu", "sigma"], kind="kde", marginals=True, divergences=True)
    plt.show()
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    ppc_values = ppc.posterior_predictive['obs'].values.flatten()*scale
    # -----------------------
    # Method 1: Standard plt.hist
    # -----------------------
    combined = np.concatenate([np.log(data), np.log(ppc_values)])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(data))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(np.log(data), bins=bins, density=True, alpha=0.5, label="Observed")
    plt.hist(np.log(ppc_values), bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(ATD)")
    plt.ylabel("Density")
    plt.title("Posterior Predictive Histogram (plt.hist)")
    plt.legend()
    plt.show()

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
    # Clean data
    df_clean = df[df['basedamage'] > 0].copy()
    df_clean = df_clean.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    
    # Prepare variables
    log_data = np.log(df_clean['basedamage'])
    ATD = df_clean['ATD'].values
    population = df_clean['population'].values
    WPC = df_clean['WPC'].values
    wind_speed = df_clean['lf_wind'].values
    pressure = df_clean['lf_pressure'].values

    with pm.Model() as model:
        # Priors for the linear combination
        alpha = pm.Normal("alpha", mu=15, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        wind_speed_coef = pm.HalfNormal("wind_speed_coef", sigma=2)
        pressure_coef = pm.Normal("pressure_coef", mu=0,sigma=2)
        
        # Linear combination of parameters (like your supervisor suggested)
        mu = (alpha )+wind_speed_coef*wind_speed+pressure_coef*pressure
        
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=log_data)
        
        trace = pm.sample(draws=2500, tune=1000, target_accept=0.95)
        summary = az.summary(trace, hdi_prob=0.95)
        print(summary)

    # Plot traces
    az.plot_trace(trace, var_names=["alpha", "sigma", "wind_speed_coef", "pressure_coef"], figsize=(12, 12))
    plt.show()
    
    az.plot_pair(trace, var_names=["alpha", "sigma", "wind_speed_coef", "pressure_coef"], 
                 kind="kde", marginals=True, divergences=True)
    plt.show()
    
    # Posterior predictive check
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    
    ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    
    # Histogram comparison
    combined = np.concatenate([log_data, ppc_values])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(log_data))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(log_data, bins=bins, density=True, alpha=0.5, label="Observed")
    plt.hist(ppc_values, bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title("Posterior Predictive Histogram")
    plt.legend()
    plt.show()
    
    return trace

if __name__ == "__main__":
    #Example usage
    df = pd.read_excel('./Speciale/Hurricane_data/Aslak_data.xls', sheet_name='ATD of ICAT', engine='xlrd')
    df_clean = df.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
    BD = np.array(df_clean['basedamage'].values)
    BD = BD[BD > 0]  #remove non-positive values

    plt.hist(np.log(BD), bins=20, density=True, alpha=0.6, color='g')
    plt.xlabel("ln(Base Damage)")
    plt.ylabel("Density")
    plt.title("Histogram of Base Damage")
    plt.savefig("./Speciale/Code/Week2/Plots/BaseDamage_histogram.png")
    plt.close()
    #plot scaled data
    plt.hist(np.log(BD/1e6), bins=20, density=True, alpha=0.6, color='b')
    plt.xlabel("ln(Scaled Base Damage)")
    plt.ylabel("Density")
    plt.title("Histogram of Scaled Base Damage")
    plt.savefig("./Speciale/Code/Week2/Plots/Scaled_BaseDamage_histogram.png")
    plt.close()
    hurricane_physical_model(df)
