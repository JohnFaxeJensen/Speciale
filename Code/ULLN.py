import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from arviz.plots import plot_utils as azpu
import pandas as pd

#Lets try to create an upper limit log normal distribution

def ulln_logp(value, mu, sigma, upper):
    safe_sigma = pt.clip(sigma, 1e-9, 1e9)
    # Log of the PDF directly
    logp = (pt.log(upper) - pt.log(safe_sigma) - pt.log(value) - pt.log(upper - value) 
            - 0.5*pt.log(2*pt.pi) - 0.5*((pt.log(value/(upper-value)) - mu)/safe_sigma)**2)
    return logp

def ulln_random(mu, sigma, upper, size=None, rng=None):
    rng = rng or np.random.default_rng()
    # Convert symbolic tensors to numeric values
    mu_val = np.array(mu).astype(float)
    sigma_val = np.array(sigma).astype(float)
    upper_val = np.array(upper).astype(float)
    
    # Sample from standard normal
    z = rng.normal(size=size, loc=mu_val, scale=sigma_val)
    # Transform to upper limit log-normal
    samples = upper_val * np.exp(z) / (1 + np.exp(z))

    
    return samples




if __name__ == "__main__":
    data = pd.read_excel(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Hurricane_data/Aslak_data.xls", sheet_name='ATD of Weinkle')
    data_clean = data.dropna(subset=['ATD'])
    atd_values = data_clean['ATD'].values
    obs = atd_values
    #plt.hist(obs, bins=15, density=True, alpha=0.5, label='Data histogram')
    #plt.show()
    upper_min = np.max(atd_values)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=4, sigma=3.0)
        sigma = pm.HalfNormal("sigma", sigma=2.0)
        
        upper = pm.Pareto("upper", alpha=2.0, m=upper_min)

        ULLN = pm.CustomDist("ULLN",
                             mu, sigma, upper,
                             logp=ulln_logp,
                             random=ulln_random,
                             observed=obs)

        trace = pm.sample(draws=1000, tune=1000, target_accept=0.95, idata_kwargs={'log_likelihood':True})
        summary = az.summary(trace, hdi_prob=0.95)
        print(summary)
        var_names = ["mu", "sigma", "upper"]
        # az.plot_trace(trace, var_names=var_names, figsize=(12, 12))
        # plt.show()
        # az.plot_pair(trace, var_names=var_names, kind="kde", marginals=True, divergences=True)
        # plt.show()
        # az.plot_posterior(trace, hdi_prob=0.95, var_names=var_names, round_to=4)
        # plt.show()
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    
    ppc_values = ppc.posterior_predictive['ULLN'].values.flatten()

    print(np.mean(np.log(ppc_values)))
    print(np.mean(np.log(obs/(upper_min+0.1 - obs))))
    plt.hist(np.log(ppc_values), bins=25, density=True, alpha=0.5, label='Posterior predictive', color='blue')
    plt.hist(np.log(obs), bins=25, density=True, alpha=0.5, label='Data histogram', color='orange')
    plt.legend()
    plt.show()
