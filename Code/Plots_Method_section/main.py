import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import norm, t, gamma, probplot
from Bayes_final import compare_models, hurricane_physical_model

# Add parent directory to path to import plot_config
sys.path.insert(0, r'C:\Users\123ti\Documents\Speciale_git\Speciale\Code')
from plot_config import set_thesis_style

# Apply consistent styling
set_thesis_style()

# make the first histogram plot with basedamage
data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\final_hurricane_data_aslak.csv")

#maybe remove data with no surge
#data = data[data['Surge_m_full_conversion_AN_as_surge'].notna()]
# Calculate bins
damage = data['basedamage'].values
log_damage = np.log(damage)
bins = int(np.round(np.sqrt(int(len(log_damage)))))

# Create side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Normal space
ax1.hist(damage, bins=bins, edgecolor='black', alpha=0.7, label='Data', density=True)
ax1.set_xlabel('Basedamage (normal scale)')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Basedamage (Normal Space)')
ax1.grid(axis='y', alpha=0.3)

ax1.legend()

# Right plot: Log space
ax2.hist(log_damage, bins=bins, edgecolor='black', alpha=0.6, color='orange', density=True, label='Data')

# Overlay normal distribution in log space
mu, sigma = norm.fit(log_damage)
print(f'Normal: μ={mu:.2f}, σ={sigma:.2f}')
x_range = np.linspace(5, 30, 200)
normal_pdf = norm.pdf(x_range, mu, sigma)
ax2.plot(x_range, normal_pdf, 'r-', linewidth=2.5, label='Normal')

# Overlay Student's t distribution in log space
df_t, loc_t, sigma_t = t.fit(log_damage)
print(f"Student's t: df={df_t:.2e}, μ={loc_t:.2f}, σ={sigma_t:.2f}")
t_pdf = t.pdf(x_range, df_t, loc_t, scale=sigma_t)
ax2.plot(x_range, t_pdf, 'b-', linewidth=2.5, label="Student's t")


ax2.set_xlabel('log(Basedamage)')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of Log(Basedamage) ')
ax2.grid(axis='y', alpha=0.3)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Plots_Method_section\basedamage_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Q-Q plot for fitted normal distribution
fig2, ax_qq = plt.subplots(figsize=(8, 6))
probplot(log_damage, dist="norm", plot=ax_qq)
ax_qq.set_title('Q-Q Plot: Log(Basedamage) vs Normal Distribution')
ax_qq.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Plots_Method_section\qq_plot_log_basedamage.png", dpi=300, bbox_inches='tight')
plt.show()


#Make model comparison plot for arviz-section
df = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Code\Data_preprocessing\generated_data\final_hurricane_data_aslak.csv")

print(df.shape)
df_clean = df.dropna(subset=['population', 'WPC', 'lf_wind', 'lf_pressure',])# 'Lat_db', 'Lon_db'])

model_specs = [
            {"exposure": True, "pressure": True, "trend": True, "surge_full": True},
            {"exposure": True, "pressure": True, "trend": True, "surge_full": True, "alpha": True},
            {"exposure": True, "trend": True, "surge_full": True, "alpha": True,"wind": True},
            ]
