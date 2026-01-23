
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    #Here is the wind models comparisons using models from the paper:
    #https://journals.ametsoc.org/view/journals/mwre/136/9/2008mwr2395.1.pdf
    
# df = pd.read_csv('./Speciale/Hurricane_data/Aslak_data_with_tide_and_travelspeed.csv')
# df_clean = df.dropna(subset=['ATD', 'population', 'WPC', 'lf_wind', 'lf_pressure'])
# df_clean = df_clean[df_clean['basedamage'] > 0]
# df_clean = df_clean[df_clean['ND'] > 0]

def equation_11_wind(pressure, lat, travel_speed):
    delta_p = 1015 - pressure
    phi = np.abs(lat)
    v_t = travel_speed  # m/s

    exponent = 0.6 * (1 - delta_p/215)
    b_s = -4.4e-5 * delta_p**2 + 0.01 * delta_p - 0.014 * phi + 0.15 * v_t**exponent + 1.0

    rho = 1.15  # kg/m³
    e = 2.71828
    vm_mps = np.sqrt((b_s / (rho * e)) * (delta_p * 100))  # delta_p in Pa
    wind_knots = vm_mps * 1.94384  # to knots
    return wind_knots
def test_pressure_to_wind_models():
    wind = df_clean['lf_wind'].values
    pressure = df_clean['lf_pressure'].values

    # --- Model calculations ---
    pressure_to_wind_version_1 = 2.3*(1010-pressure)**0.76*1.94384  # Eq (4) Knaff & Zehr
    pressure_to_wind_version_2 = 3.92*(1015-pressure)**0.644*1.94384  # Eq (3) Dvorak Atlantic

    # Simplified Eq (11)
    delta_p = 1015 - pressure
    phi = np.abs(df_clean['lf_lat'].values)
    v_t = df_clean['travel_speed_after_landfall_m_s'].values  # m/s

    # Note: The exponent in the paper is: 0.6*(1 - delta_p/215)
    # Let's compute it correctly
    exponent = 0.6 * (1 - delta_p/215)
    b_s = -4.4e-5 * delta_p**2 + 0.01 * delta_p - 0.014 * phi + 0.15 * v_t**exponent + 1.0

    rho = 1.15  # kg/m³
    e = 2.71828
    vm_mps = np.sqrt((b_s / (rho * e)) * (delta_p * 100))  # delta_p in Pa
    pressure_to_wind_version_3 = vm_mps * 1.94384  # to knots

    # --- Metrics calculation ---
    models = {
        'Knaff & Zehr (Eq 4)': pressure_to_wind_version_1,
        'Dvorak Atlantic (Eq 3)': pressure_to_wind_version_2,
        'Simplified Eq (11)': pressure_to_wind_version_3
    }

    print("=== Model Performance Metrics ===")
    for name, modeled in models.items():
        bias = np.mean(modeled - wind)
        mae = mean_absolute_error(wind, modeled)
        rmse = np.sqrt(mean_squared_error(wind, modeled))
        r2 = r2_score(wind, modeled)
        
        print(f"\n{name}:")
        print(f"  Bias: {bias:.2f} kt")
        print(f"  MAE:  {mae:.2f} kt")
        print(f"  RMSE: {rmse:.2f} kt")
        print(f"  R²:   {r2:.3f}")
        #save metrics to a file
        df = pd.DataFrame({
            'Model': [name],
            'Bias (kt)': [bias],
            'MAE (kt)': [mae],
            'RMSE (kt)': [rmse],
            'R²': [r2]
        })
        df.to_csv(f"./Speciale/Code/Week4/Plots/{name.replace(' ', '_')}_metrics.csv", index=False)

    # --- 1. Scatter plot: Observed vs Modeled ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(wind, pressure_to_wind_version_1, alpha=0.6, color='b', label='Knaff & Zehr (Eq 4)', s=20)
    plt.scatter(wind, pressure_to_wind_version_2, alpha=0.6, color='g', label='Dvorak Atlantic (Eq 3)', s=20)
    plt.scatter(wind, pressure_to_wind_version_3, alpha=0.6, color='r', label='Simplified Eq (11)', s=20)
    plt.plot([0, 200], [0, 200], 'k--', linewidth=1)
    plt.xlabel("Observed Wind Speed (knots)")
    plt.ylabel("Modeled Wind Speed (knots)")
    plt.title("Observed vs Modeled Wind Speed")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- 2. Residuals plot: Modeled - Observed vs Pressure ---
    plt.subplot(1, 2, 2)
    residuals1 = pressure_to_wind_version_1 - wind
    residuals2 = pressure_to_wind_version_2 - wind
    residuals3 = pressure_to_wind_version_3 - wind

    plt.scatter(pressure, residuals1, alpha=0.6, color='b', label='Knaff & Zehr', s=20)
    plt.scatter(pressure, residuals2, alpha=0.6, color='g', label='Dvorak Atlantic', s=20)
    plt.scatter(pressure, residuals3, alpha=0.6, color='r', label='Simplified Eq (11)', s=20)

    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.xlabel("Central Pressure (mb)")
    plt.ylabel("Modeled - Observed Wind (knots)(residuals)")
    plt.title("Residuals vs Central Pressure")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # Stronger storms (lower pressure) to the right

    plt.tight_layout()
    plt.savefig("./Speciale/Code/Week4/Plots/Pressure_to_Wind_comparison_with_metrics.png", dpi=150)
    plt.show()