import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# True parameters that we'll try to recover

TRUE_SLOPE = 2.5
TRUE_INTERCEPT = 1.0
TRUE_SIGMA = 0.5

# Generate data
n_points = 50
x_data = np.linspace(0, 10, n_points)
y_data = TRUE_INTERCEPT + TRUE_SLOPE * x_data + np.random.normal(0, TRUE_SIGMA, n_points)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.7, label='Observed data')
plt.plot(x_data, TRUE_INTERCEPT + TRUE_SLOPE * x_data, 'r-', linewidth=2, 
         label=f'True line: y = {TRUE_INTERCEPT} + {TRUE_SLOPE}x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Our Fake Data with True Relationship')
plt.show()

print(f"True parameters: intercept={TRUE_INTERCEPT}, slope={TRUE_SLOPE}, sigma={TRUE_SIGMA}")

def likelihood(y_obs, y_pred, sigma):
    """
    Calculate the likelihood: P(data | parameters)
    How probable is our observed data given these parameters?
    """
    # Normal distribution likelihood
    log_lik = np.sum(stats.norm.logpdf(y_obs, loc=y_pred, scale=sigma))
    return log_lik

def prior(intercept, slope, sigma):
    """
    Calculate the prior: P(parameters)
    How plausible are these parameters before seeing data?
    """
    # Weakly informative priors
    log_prior = 0
    
    # Intercept: normal prior around 0
    log_prior += stats.norm.logpdf(intercept, loc=0, scale=10)
    
    # Slope: normal prior around 0  
    log_prior += stats.norm.logpdf(slope, loc=0, scale=10)
    
    # Sigma: half-normal prior (must be positive)
    log_prior += stats.halfnorm.logpdf(sigma, scale=5)
    
    return log_prior

def posterior(y_obs, x_obs, intercept, slope, sigma):
    """
    Calculate the posterior: P(parameters | data) ∝ P(data | parameters) × P(parameters)
    """
    y_pred = intercept + slope * x_obs
    log_lik = likelihood(y_obs, y_pred, sigma)
    log_prior = prior(intercept, slope, sigma)
    return log_lik + log_prior  # Log posterior

def metropolis_hastings(y_obs, x_obs, n_samples=10000, burn_in=1000):
    """
    Our custom MCMC implementation!
    """
    # Initialize chains
    intercept_chain = np.zeros(n_samples + burn_in)
    slope_chain = np.zeros(n_samples + burn_in)
    sigma_chain = np.zeros(n_samples + burn_in)
    
    # Starting values (bad guesses on purpose!)
    intercept_chain[0] = -5.0
    slope_chain[0] = -1.0  
    sigma_chain[0] = 3.0
    
    # Proposal distribution step sizes
    proposal_sd = {'intercept': 0.2, 'slope': 0.1, 'sigma': 0.1}
    
    # Acceptance counters
    acceptances = {'intercept': 0, 'slope': 0, 'sigma': 0}
    
    print("Running MCMC sampling...")
    for i in tqdm(range(1, n_samples + burn_in)):
        current_intercept = intercept_chain[i-1]
        current_slope = slope_chain[i-1]
        current_sigma = sigma_chain[i-1]
        
        # Current posterior value
        current_post = posterior(y_obs, x_obs, current_intercept, current_slope, current_sigma)
        
        # Sample each parameter one at a time (Gibbs-style within Metropolis)
        
        # 1. Update intercept
        proposed_intercept = np.random.normal(current_intercept, proposal_sd['intercept'])
        proposed_post = posterior(y_obs, x_obs, proposed_intercept, current_slope, current_sigma)
        
        acceptance_ratio = np.exp(proposed_post - current_post)
        if np.random.random() < acceptance_ratio:
            intercept_chain[i] = proposed_intercept
            acceptances['intercept'] += 1
        else:
            intercept_chain[i] = current_intercept
        
        # 2. Update slope  
        current_intercept = intercept_chain[i]  # Use latest value
        proposed_slope = np.random.normal(current_slope, proposal_sd['slope'])
        proposed_post = posterior(y_obs, x_obs, current_intercept, proposed_slope, current_sigma)
        
        acceptance_ratio = np.exp(proposed_post - current_post)
        if np.random.random() < acceptance_ratio:
            slope_chain[i] = proposed_slope
            acceptances['slope'] += 1
        else:
            slope_chain[i] = current_slope
        
        # 3. Update sigma
        current_slope = slope_chain[i]  # Use latest value
        proposed_sigma = np.random.normal(current_sigma, proposal_sd['sigma'])
        # Sigma must be positive!
        if proposed_sigma > 0:
            proposed_post = posterior(y_obs, x_obs, current_intercept, current_slope, proposed_sigma)
            acceptance_ratio = np.exp(proposed_post - current_post)
            if np.random.random() < acceptance_ratio:
                sigma_chain[i] = proposed_sigma
                acceptances['sigma'] += 1
            else:
                sigma_chain[i] = current_sigma
        else:
            sigma_chain[i] = current_sigma
    
    # Calculate acceptance rates
    total_iterations = n_samples + burn_in - 1
    acceptance_rates = {param: count/total_iterations for param, count in acceptances.items()}
    
    print(f"\nAcceptance rates: {acceptance_rates}")
    
    # Remove burn-in period
    intercept_chain = intercept_chain[burn_in:]
    slope_chain = slope_chain[burn_in:]
    sigma_chain = sigma_chain[burn_in:]
    
    return intercept_chain, slope_chain, sigma_chain


# Run our sampler
intercept_samples, slope_samples, sigma_samples = metropolis_hastings(
    y_data, x_data, n_samples=15000, burn_in=1000
)

print(f"Collected {len(intercept_samples)} samples after burn-in")

# Trace plots to check convergence
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Intercept trace and distribution
axes[0, 0].plot(intercept_samples, alpha=0.7)
axes[0, 0].axhline(y=TRUE_INTERCEPT, color='red', linestyle='--', label='True value')
axes[0, 0].set_ylabel('Intercept')
axes[0, 0].legend()

axes[0, 1].hist(intercept_samples, bins=30, alpha=0.7, density=True)
axes[0, 1].axvline(x=TRUE_INTERCEPT, color='red', linestyle='--', label='True value')
axes[0, 1].axvline(x=np.mean(intercept_samples), color='blue', linestyle='-', label='Posterior mean')
axes[0, 1].set_xlabel('Intercept')
axes[0, 1].legend()

# Slope trace and distribution
axes[1, 0].plot(slope_samples, alpha=0.7)
axes[1, 0].axhline(y=TRUE_SLOPE, color='red', linestyle='--', label='True value')
axes[1, 0].set_ylabel('Slope')

axes[1, 1].hist(slope_samples, bins=30, alpha=0.7, density=True)
axes[1, 1].axvline(x=TRUE_SLOPE, color='red', linestyle='--', label='True value')
axes[1, 1].axvline(x=np.mean(slope_samples), color='blue', linestyle='-', label='Posterior mean')
axes[1, 1].set_xlabel('Slope')

# Sigma trace and distribution
axes[2, 0].plot(sigma_samples, alpha=0.7)
axes[2, 0].axhline(y=TRUE_SIGMA, color='red', linestyle='--', label='True value')
axes[2, 0].set_ylabel('Sigma')

axes[2, 1].hist(sigma_samples, bins=30, alpha=0.7, density=True)
axes[2, 1].axvline(x=TRUE_SIGMA, color='red', linestyle='--', label='True value')
axes[2, 1].axvline(x=np.mean(sigma_samples), color='blue', linestyle='-', label='Posterior mean')
axes[2, 1].set_xlabel('Sigma')

plt.tight_layout()
plt.suptitle('MCMC Results - Trace Plots and Posterior Distributions', y=1.02)
plt.show()

# Calculate posterior summaries
def calculate_summary(samples, true_value, name):
    mean = np.mean(samples)
    std = np.std(samples)
    hdi_lower = np.percentile(samples, 2.5)
    hdi_upper = np.percentile(samples, 97.5)
    
    print(f"{name:10s}: True = {true_value:6.3f} | "
          f"Posterior = {mean:6.3f} ± {std:5.3f} | "
          f"94% HDI = [{hdi_lower:5.3f}, {hdi_upper:5.3f}] | "
          f"Error = {abs(mean - true_value):.3f}")

print("\n" + "="*80)
print("POSTERIOR SUMMARY vs TRUE VALUES")
print("="*80)
calculate_summary(intercept_samples, TRUE_INTERCEPT, "Intercept")
calculate_summary(slope_samples, TRUE_SLOPE, "Slope")
calculate_summary(sigma_samples, TRUE_SIGMA, "Sigma")

# Posterior predictive check
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, alpha=0.7, label='Observed data')

# Plot true relationship
plt.plot(x_data, TRUE_INTERCEPT + TRUE_SLOPE * x_data, 'r-', linewidth=3, 
         label='True relationship')

# Plot posterior mean relationship
posterior_intercept = np.mean(intercept_samples)
posterior_slope = np.mean(slope_samples)
plt.plot(x_data, posterior_intercept + posterior_slope * x_data, 'b--', linewidth=2,
         label='Posterior mean')

# Plot several posterior samples
for i in range(100):
    idx = np.random.randint(len(intercept_samples))
    plt.plot(x_data, intercept_samples[idx] + slope_samples[idx] * x_data, 
             'gray', alpha=0.05)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Posterior Predictive Check')
plt.show()