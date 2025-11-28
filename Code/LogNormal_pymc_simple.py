import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt    
import scipy.stats as stats
import arviz as az
from matplotlib import pyplot as plt
import os

def test_standard_lognormal(ATD, threshold=0.5):
    #make plot dir if not exists
    path = "./Speciale/Code/Plots/Simulated_posteriors/No_transform/Threshold_{}".format(threshold)
    os.makedirs(path, exist_ok=True)
    #infer on transformed data
    original_size = len(ATD)
    ATD_filtered = ATD[ATD > threshold]  # Filter data below the threshold
    new_size = len(ATD_filtered)
    points_removed = original_size - new_size
    log_ATD = np.log(ATD_filtered)
    log_mean = np.mean(log_ATD)
    log_std = np.std(log_ATD)

    #Define the model, mu and sigma are the parameters of the underlying normal distribution, prior based on data
    #pm.lognormal uses mu and sigma of the underlying normal distribution and is the likelihood
    
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=18, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=6)

        lognormal_obs = pm.LogNormal('obs', mu=mu, sigma=sigma, observed=ATD_filtered)
    #Run the model
    with model:
        # Sample from the posterior.
        #This function call performs the MCMC sampling to generate samples from the posterior distribution 
        # of the model parameters. It uses 2000 tuning steps to adapt the sampler and then collects 5,000 samples
        trace = pm.sample(2500, tune=1000, target_accept=0.95, random_seed=42)
        #It works by exploring the parameter space using No-U-Turn Sampler (NUTS), an efficient variant of Hamiltonian Monte Carlo (HMC).
        #This means that 
        summary = az.summary(trace, hdi_prob=0.95)
        summary.to_csv(path + "/posterior_summary.csv")

    

    az.plot_trace(trace, var_names=["mu", "sigma"], figsize=(12, 12))
    plt.savefig(path + "/trace_plot.png")
    plt.close()
    az.plot_pair(trace, var_names=["mu", "sigma"], kind="kde", marginals=True, divergences=True)
    plt.savefig(path + "/pair_plot.png")
    plt.close()
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    ppc_values = ppc.posterior_predictive['obs'].values.flatten()
    print(len(ppc_values))
    # -----------------------
    # Method 1: Standard plt.hist
    # -----------------------
    combined = np.concatenate([np.log(ATD_filtered), np.log(ppc_values)])
    bins = np.linspace(combined.min(), combined.max(), max(int(np.sqrt(len(ATD_filtered))), 25))
    plt.figure(figsize=(10,6))
    plt.hist(np.log(ATD_filtered), bins=bins, density=True, alpha=0.5, label="Observed")
    plt.hist(np.log(ppc_values), bins=bins, density=True, alpha=0.5, label="Posterior predictive")
    #plt.xscale('log')
    plt.xlabel("ln(ATD)")
    plt.ylabel("Density")
    plt.title(f"Posterior Predictive Histogram (plt.hist), Threshold {threshold}, Points removed: {points_removed}")
    plt.legend()
    plt.savefig(os.path.join(path, "simulated_posterior_hist.png"))
    plt.close()

    # # -----------------------
    # # Step plot of proportion per bin
    # # -----------------------

    hist_obs, edges = np.histogram(np.log(ATD_filtered), bins=bins)
    hist_ppc, _ = np.histogram(np.log(ppc_values), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    plt.figure(figsize=(10,6))
    plt.step(centers, hist_obs / len(ATD_filtered), where='mid', label="Observed (>threshold) proportion per bin")
    plt.step(centers, hist_ppc / len(ppc_values), where='mid', label="Posterior predictive proportion per bin")

    plt.xlabel("ln(ATD)")
    plt.ylabel("Proportion per bin")
    plt.title(f"Posterior Predictive Histogram (step plot), Threshold {threshold}, Points removed: {points_removed}")
    plt.legend()
    plt.savefig(os.path.join(path, "simulated_posterior_step.png"))
    plt.close()




if __name__ == "__main__":
    df = pd.read_excel('./Speciale/Hurricane_data/Aslak_data.xls', sheet_name='ATD of ICAT', engine='xlrd')

    ATD = df['ATD'].values

    ATD = ATD[ATD > 0] 
    full_min = ATD.min()
    full_max = ATD.max()

    
    #ATD = ATD[ATD < 200] 
    plt.hist(np.log(ATD), bins=int(np.sqrt(len(ATD))), density=True, alpha=0.6, color='g')
    plt.title("Histogram of Raw ICAT ATD Data")
    plt.xlabel("ln(ATD)")
    plt.ylabel("Density")
    plt.savefig("./Speciale/Code/Plots/Raw_ICAT_Hist.png")
    plt.close()
    test_standard_lognormal(ATD, threshold=0)
    
    
    






