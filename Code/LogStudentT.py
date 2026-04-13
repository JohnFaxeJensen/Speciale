import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import gammaln

# ============================================================================
# LOG-STUDENT'S T DISTRIBUTION (CustomDist for PyMC)
# ============================================================================
# Based on the log-t distribution PDF:
# p(x | ν, μ̂, σ̂) = Γ((ν+1)/2) / (x·Γ(ν/2)·√(πνσ)) · 
#                    (1 + 1/ν·((ln x - μ̂)/σ̂)²)^(-(ν+1)/2)

def log_studentt_logp(value, mu, sigma, nu):
    """
    Log probability density for log-Student's t distribution.
    
    Parameters
    ----------
    value : tensor
        Observed data (on original, positive scale)
    mu : tensor
        Location parameter (on log scale)
    sigma : tensor
        Scale parameter (on log scale, > 0)
    nu : tensor
        Degrees of freedom (> 0). Controls tail heaviness:
        - nu < 5: heavy tails
        - nu = 5-10: moderate tails
        - nu > 20: approaches normal distribution
    
    Returns
    -------
    logp : tensor
        Log probability density
    
    Notes
    -----
    The distribution is derived from Student's t on log scale.
    If log(X) ~ t(mu, sigma, nu), then X ~ LogStudentT(mu, sigma, nu)
    """
    # Numerical stability
    safe_value = pt.clip(value, 1e-10, np.inf)
    safe_sigma = pt.clip(sigma, 1e-9, 1e9)
    safe_nu = pt.clip(nu, 1.0, 1e6)
    
    # Log transform
    log_value = pt.log(safe_value)
    
    # Standardized residual: z = (ln(x) - μ) / σ
    z = (log_value - mu) / safe_sigma
    z_squared = z ** 2
    
    # Log probability using the PDF formula:
    # ln p(x) = ln Γ((ν+1)/2) - ln Γ(ν/2) - ln(x) 
    #           - 1/2·ln(πνσ²) - (ν+1)/2·ln(1 + z²/ν)
    
    logp = (
        pt.gammaln((safe_nu + 1) / 2.0) 
        - pt.gammaln(safe_nu / 2.0)
        - pt.log(safe_value)
        - 0.5 * pt.log(np.pi * safe_nu * safe_sigma ** 2)
        - (safe_nu + 1.0) / 2.0 * pt.log(1.0 + z_squared / safe_nu)
    )
    
    return logp


def log_studentt_random(mu, sigma, nu, size=None, rng=None):
    """
    Random sampling from log-Student's t distribution.
    
    Parameters
    ----------
    mu : float or array
        Location parameter (on log scale)
    sigma : float or array
        Scale parameter (on log scale, > 0)
    nu : float or array
        Degrees of freedom (> 0)
    size : int or tuple, optional
        Shape of samples
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.
    
    Returns
    -------
    samples : ndarray
        Samples from log-Student's t distribution (on original scale)
    
    Method
    ------
    1. Sample z ~ StandardStudent's t(nu)
    2. Transform: y = mu + sigma * z  (this is Student's t on log scale)
    3. Exponentiate: x = exp(y)  (log-Student's t on original scale)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Convert inputs to float arrays
    mu_val = np.atleast_1d(np.asarray(mu, dtype=float))
    sigma_val = np.atleast_1d(np.asarray(sigma, dtype=float))
    nu_val = np.atleast_1d(np.asarray(nu, dtype=float))
    
    # Determine output size
    if size is None:
        # Scalar-like inputs -> scalar output
        if (mu_val.shape == (1,) and sigma_val.shape == (1,) and nu_val.shape == (1,)):
            size = 1
        else:
            size = max(mu_val.size, sigma_val.size, nu_val.size)
    else:
        size = int(np.prod(size)) if isinstance(size, tuple) else int(size)
    
    # Broadcast to common shape
    mu_val = np.broadcast_to(mu_val, (size,))
    sigma_val = np.broadcast_to(sigma_val, (size,))
    nu_val = np.broadcast_to(nu_val, (size,))
    
    # Sample from standard Student's t (chi-squared / nu method)
    # Student's t(nu) = Normal(0,1) / sqrt(ChiSq(nu) / nu)
    z_normal = rng.standard_normal(size)
    chi_sq = rng.chisquare(nu_val)
    z_studentt = z_normal / np.sqrt(chi_sq / nu_val)
    
    # Transform to Student's t(mu, sigma, nu) on log scale
    y = mu_val + sigma_val * z_studentt
    
    # Exponentiate to original scale
    samples = np.exp(y)
    
    return samples


