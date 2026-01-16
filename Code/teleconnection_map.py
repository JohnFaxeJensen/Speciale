import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats

def create_teleconnection_map_icoads(damage_df, icoads_path, output_dir=r"./Speciale/Code/Week4/Plots"):
    """
    Create a teleconnection map using ICOADS (observed) data: 
    correlation between yearly log(damage) and ICOADS SST for each pixel.
    
    Parameters:
    -----------
    damage_df : DataFrame
        Hurricane damage data with columns: 'Year', 'basedamage' (or 'ATD'/'ND')
    icoads_path : str
        Path to ICOADS SST NetCDF file
    output_dir : str
        Directory to save output plots
    
    Returns:
    --------
    correlation_map : xarray.DataArray
        Correlation coefficient for each pixel
    pvalue_map : xarray.DataArray
        P-value for each pixel
    """
    
    # 1. Sum log(damage) by year
    df_yearly = damage_df[damage_df['basedamage'] > 0].copy()
    df_yearly['log_damage'] = np.log(df_yearly['basedamage'])
    yearly_damage = df_yearly.groupby('Year')['log_damage'].sum().reset_index()
    yearly_damage.columns = ['Year', 'total_log_damage']
    
    print(f"Yearly damage summary:")
    print(yearly_damage)
    
    # 2. Load ICOADS data
    print(f"\nLoading ICOADS from: {icoads_path}")
    ds = xr.open_dataset(icoads_path)
    sst_data = ds['sst']  # Shape: (time, lat, lon)
    
    # Extract years from time coordinate
    years_in_data = pd.to_datetime(sst_data.time.values).year
    
    # 3. Compute yearly mean SST from monthly data
    sst_data['year'] = ('time', years_in_data)
    sst_yearly = sst_data.groupby('year').mean(dim='time')
    
    print(f"\nICOADS shape: {sst_yearly.shape}")
    print(f"Year range in ICOADS data: {sst_yearly.year.min().values} - {sst_yearly.year.max().values}")
    
    # 4. Align years between damage and SST data
    common_years = np.intersect1d(
        yearly_damage['Year'].values,
        sst_yearly.year.values
    )
    
    print(f"Common years: {len(common_years)} (from {common_years[0]} to {common_years[-1]})")
    
    # Select common years
    yearly_damage_aligned = yearly_damage[yearly_damage['Year'].isin(common_years)].sort_values('Year')
    sst_aligned = sst_yearly.sel(year=common_years)
    
    damage_values = yearly_damage_aligned['total_log_damage'].values
    
    # 5. Calculate correlation at each pixel
    nlat, nlon = sst_aligned.shape[1], sst_aligned.shape[2]
    correlation_map = np.zeros((nlat, nlon))
    pvalue_map = np.zeros((nlat, nlon))
    
    print(f"\nCalculating correlations for {nlat} x {nlon} = {nlat*nlon} pixels...")
    
    for i in range(nlat):
        if i % 5 == 0:
            print(f"  Processing latitude {i}/{nlat}")
        for j in range(nlon):
            sst_pixel = sst_aligned.values[:, i, j]
            
            # Skip if all NaN
            if np.isnan(sst_pixel).all():
                correlation_map[i, j] = np.nan
                pvalue_map[i, j] = np.nan
            else:
                # Compute Pearson correlation
                valid_mask = ~(np.isnan(sst_pixel) | np.isnan(damage_values))
                if valid_mask.sum() > 2:  # Need at least 3 points
                    corr, pval = stats.pearsonr(
                        damage_values[valid_mask],
                        sst_pixel[valid_mask]
                    )
                    correlation_map[i, j] = corr
                    pvalue_map[i, j] = pval
                else:
                    correlation_map[i, j] = np.nan
                    pvalue_map[i, j] = np.nan
    
    # 6. Create xarray DataArray for easier handling
    correlation_map = xr.DataArray(
        correlation_map,
        coords={
            'latitude': sst_aligned.lat.values,
            'longitude': sst_aligned.lon.values
        },
        dims=['latitude', 'longitude'],
        name='correlation'
    )
    
    pvalue_map = xr.DataArray(
        pvalue_map,
        coords={
            'latitude': sst_aligned.lat.values,
            'longitude': sst_aligned.lon.values
        },
        dims=['latitude', 'longitude'],
        name='pvalue'
    )
    
    # 7. Create visualization
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot correlation with better colormap
    levels = np.linspace(-1, 1, 21)
    cf = ax.contourf(
        correlation_map.longitude,
        correlation_map.latitude,
        correlation_map,
        levels=levels,
        cmap='RdBu_r',
        transform=ccrs.PlateCarree(),
        extend='both'
    )
    
    # Mask non-significant correlations (p > 0.05)
    sig_mask = pvalue_map.values > 0.05
    significant_corr = correlation_map.copy()
    significant_corr.values[sig_mask] = np.nan
    
    # Overlay significant correlations with hatching
    ax.contourf(
        significant_corr.longitude,
        significant_corr.latitude,
        significant_corr,
        levels=levels,
        hatches=['///', None, '\\\\\\'],
        alpha=0.3,
        transform=ccrs.PlateCarree()
    )
    
    # Add coastlines and features
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Atlantic plus gulf
    ax.set_extent([-100, 30, -20, 60], crs=ccrs.PlateCarree())
    
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Correlation with Annual log(Damage)', fontsize=12)
    
    plt.title(f'Teleconnection Map: ICOADS SST vs Hurricane Damage (Observed Data)\n' + 
              f'Years: {common_years[0]}-{common_years[-1]}, N={len(common_years)}',
              fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/teleconnection_map_icoads_damage.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTeleconnection map saved to: {output_dir}/teleconnection_map_icoads_damage.png")
    
    # 8. Additional statistics
    print(f"\nCorrelation statistics (ICOADS):")
    print(f"  Mean correlation: {np.nanmean(correlation_map.values):.3f}")
    print(f"  Std correlation: {np.nanstd(correlation_map.values):.3f}")
    print(f"  Max correlation: {np.nanmax(correlation_map.values):.3f}")
    print(f"  Min correlation: {np.nanmin(correlation_map.values):.3f}")
    print(f"  Significant pixels (p<0.05): {(pvalue_map.values < 0.05).sum()}")
    
    return correlation_map, pvalue_map


if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('./Speciale/Hurricane_data/Aslak_data_with_tide_and_travelspeed.csv')
    
    # Load and merge temperature data if needed
    df_temp_data = pd.read_csv('./Speciale/temp_data/data_global_ocean_temp.csv', skiprows=3)
    from Speciale.Code.merge_temp import merge_temp_data, merge_temp_data_monthly
    df = merge_temp_data(df, df_temp_data, 'Year')
    
    # Create teleconnection map using ICOADS (observed data)
    correlation_map, pvalue_map = create_teleconnection_map_icoads(
        damage_df=df,
        icoads_path='./Speciale/temp_data/sst_monthly_mean_icoads.nc',
        output_dir=r"./Speciale/Code/Week4/Plots"
    )
    
    # # Save maps as NetCDF
    # correlation_map.to_netcdf(r"./Speciale/Code/Week4/Plots/teleconnection_correlation_icoads.nc")
    # pvalue_map.to_netcdf(r"./Speciale/Code/Week4/Plots/teleconnection_pvalue_icoads.nc")