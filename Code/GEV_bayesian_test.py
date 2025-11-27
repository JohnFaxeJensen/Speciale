import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import genextreme
from arviz.plots import plot_utils as azpu


def gev_logp(value, mu, sigma, xi):

    # Support: sigma > 0
    safe_sigma = pt.clip(sigma, 1e-9, 1e9)

    # Compute standardized variable
    t = (value - mu) / safe_sigma
    z = 1 + xi * t

    # GEV support condition
    # logp = -inf when z <= 0
    logp = pm.math.switch(
        pt.le(z, 0),
        -np.inf,
        -(1/xi + 1) * pt.log(z) - z ** (-1/xi) - pt.log(safe_sigma)
    )

    # Special case: xi ≈ 0 gives Gumbel distribution
    logp = pm.math.switch(
        pt.abs(xi) < 1e-6,
        -(t + pt.exp(-t)) - pt.log(safe_sigma),
        logp
    )

    return logp

def gev_random(mu, sigma, xi, size=None, rng=None):
    rng = rng or np.random.default_rng()
    # Convert symbolic tensors to numeric values
    mu_val = np.array(mu).astype(float)
    sigma_val = np.array(sigma).astype(float)
    xi_val = np.array(xi).astype(float)
    

    return genextreme.rvs(c=-xi_val, loc=mu_val, scale=sigma_val, size=size, random_state=rng)


if __name__ == "__main__":
    data = np.array([4.03, 3.83, 3.65, 3.88, 4.01, 4.08, 4.18, 3.80, 
                 4.36, 3.96, 3.98, 4.69, 3.85, 3.96, 3.85, 3.93, 
                 3.75, 3.63, 3.57, 4.25, 3.97, 4.05, 4.24, 4.22, 
                 3.73, 4.37, 4.06, 3.71, 3.96, 4.06, 4.55, 3.79, 
                 3.89, 4.11, 3.85, 3.86, 3.86, 4.21, 4.01, 4.11, 
                 4.24, 3.96, 4.21, 3.74, 3.85, 3.88, 3.66, 4.11, 
                 3.71, 4.18, 3.90, 3.78, 3.91, 3.72, 4.00, 3.66, 
                 3.62, 4.33, 4.55, 3.75, 4.08, 3.90, 3.88, 3.94, 
                 4.33])
    with pm.Model() as model:
        μ = pm.Normal("μ", mu=3.8, sigma=0.2)
        σ = pm.HalfNormal("σ", sigma=0.3)
        ξ = pm.TruncatedNormal("ξ", mu=0, sigma=0.2, lower=-0.6, upper=0.6)

        gev = pm.CustomDist(
            "gev",
            μ, σ, ξ,
            logp=gev_logp,
            random=gev_random,
            observed=data
        )
        z_p = pm.Deterministic("z_p", μ - σ / ξ * (1 - (-np.log(1 - 0.1)) ** (-ξ)))
    idata = pm.sample_prior_predictive(samples=1000, model=model)
    az.plot_ppc(idata, group="prior", figsize=(12, 6))
    ax = plt.gca()
    ax.set_xlim([2, 6])
    ax.set_ylim([0, 2])
    plt.show()
    az.plot_posterior(
    idata, group="prior", var_names=["μ", "σ", "ξ"], hdi_prob="hide", point_estimate=None)
    plt.show()
    with model:
        trace = pm.sample(
            5000,
            cores=4,
            chains=4,
            tune=2000,
            initvals={"μ": -0.5, "σ": 1.0, "ξ": -0.1},
            target_accept=0.98,
        )
    # add trace to existing idata object
    idata.extend(trace)
    az.plot_trace(idata, var_names=["μ", "σ", "ξ"], figsize=(12, 12))
    plt.show()
    az.hdi(idata, hdi_prob=0.95)
    az.plot_posterior(idata, hdi_prob=0.95, var_names=["z_p"], round_to=4)
    plt.show()
    az.plot_dist_comparison(idata, var_names=["z_p"])
    plt.show()
    az.plot_pair(idata, var_names=["μ", "σ", "ξ"], kind="kde", marginals=True, divergences=True)
    plt.show()

