import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import pandas as pd
import datetime as dt
import geopandas as gpd
from shapely.geometry import Polygon, box
import xarray as xr
import numpy as np
import regionmask

gulf = gpd.read_file(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temperature_regions/gulf_of_mexico/iho.shp")
caribbean = gpd.read_file(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temperature_regions/caribbean_sea/iho.shp")
# Ensure geographic CRS (lon/lat)

gulf = gulf.to_crs(epsg=4326)
caribbean = caribbean.to_crs(epsg=4326)
merged = gpd.GeoDataFrame(pd.concat([gulf, caribbean]), crs="EPSG:4326")

cut_box = Polygon([
    (-83.0, 15.0),   # bottom-left
    (-60.0, 15.0),   # bottom-right
    (-60.0, 23.0),   # right side (cut starts here)
    (-75.0, 33.0),   # diagonal cut point
    (-83.0, 33.0),   # top-left
])

gulf_carib_geom = merged.geometry.union_all()
cut_box = gpd.GeoSeries(cut_box, crs="EPSG:4326")

gulf_carib_merged = gulf_carib_geom.union(cut_box)
#transform back to polygon
gulf_carib_merged = gulf_carib_merged
# Atlantic MDR as geometry (box)
mdr_geom = box(-59.5, 9.0, -15.0, 25.0)

# Unified regions dictionary with geometries
regions = {
    "Gulf + Caribbean": {
        "geometry": gulf_carib_merged[0], #polygon,
        "color": "tab:blue",
    },
    "Atlantic MDR": {
        "geometry": mdr_geom,
        "color": "tab:red",
    },
}



def load_genesis_points(csv_path):
    # Load IBTrACS filtered CSV and compute genesis (first observation per storm)
    df = pd.read_csv(csv_path)
    # Use USA_ATCF_ID to identify storms; parse time
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df = df.dropna(subset=["ISO_TIME", "USA_ATCF_ID", "LAT", "LON"])
    # Sort and take first per USA_ATCF_ID
    df_sorted = df.sort_values(["USA_ATCF_ID", "ISO_TIME"])  # earliest first
    genesis = df_sorted.groupby("USA_ATCF_ID", as_index=False).first()
    # Return only lat/lon
    return genesis[["LAT", "LON", "USA_ATCF_ID", "ISO_TIME"]]


def plot_regions_with_genesis(regions_dict, genesis_df, out_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-98, -15, 5, 35], crs=ccrs.PlateCarree())

    # Basemap
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.3)
    gl.right_labels = False
    gl.top_labels = False

    # Draw regions from geometries
    for name, r in regions_dict.items():
        geom = r["geometry"]
        color = r.get("color", "black")
        
        # Add geometry
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor=color,
            linewidth=1.6
        )
        
        # Label at centroid or top-left of bounds
        bounds = geom.bounds  # (minx, miny, maxx, maxy)

        ax.text(
            bounds[0] + 0.5,
            bounds[3] - 0.5,
            name,
            fontsize=10,
            color=color,
            transform=ccrs.PlateCarree(),
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
        )

    # Genesis points
    ax.scatter(
        genesis_df["LON"], genesis_df["LAT"],
        s=12, color="tab:orange", alpha=0.8, edgecolors="none",
        transform=ccrs.PlateCarree(), label="Genesis positions",
    )

    ax.set_title("Genesis positions with Gulf/Caribbean + Atlantic MDR regions")
    ax.legend(loc="lower left")
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()


def process_sst_dataset(sst, regions_dict, dataset_name, sample_time, output_csv, suffix, 
                        lon_name='longitude', lat_name='latitude', x_plot='longitude', y_plot='latitude'):

    mean_temps_all = []
    for name, r in regions_dict.items():
        geom = r["geometry"]
        latitude = sst[lat_name]
        longitude = sst[lon_name]
        
        mask = regionmask.Regions(
            [geom],
            names=[name],
            abbrevs=[name[:3]]
        )
        mask_2d = mask.mask(longitude, latitude)
        sst_region = sst.where(mask_2d == 0)
        
        # Calculate mean temp for region for each month
        mean_temps = []
        for time in sst_region.time.values:
            subset = sst_region.sel(time=time)
            mean_temp = subset.mean().values
            year = pd.to_datetime(time).year
            month = pd.to_datetime(time).month
            mean_temps.append({'Year': year, 'Month': month, f'mean_temp_{suffix}': mean_temp, 'region': name})
        
        mean_temps_all.append({'region': name, 'data': mean_temps})
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={"projection": ccrs.PlateCarree()})
        sst_region.sel(time=sample_time).plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            x=x_plot,
            y=y_plot,
            cmap="coolwarm",
            add_colorbar=True
        )
        
        gpd.GeoSeries(r["geometry"]).boundary.plot(ax=ax, edgecolor="black", linewidth=2)
        ax.set_extent([-98, -15, 5, 35], crs=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(f"SST + Region outline ({name}, {sample_time}) - {dataset_name}")
        plt.show()
    
    # Save to CSV
    gc_df = pd.DataFrame(mean_temps_all[0]['data'])
    mdr_df = pd.DataFrame(mean_temps_all[1]['data'])
    merged_df = pd.merge(gc_df, mdr_df, on=['Year', 'Month'], suffixes=('_gc', '_mdr'))
    merged_df.to_csv(output_csv, index=False)
    
    return merged_df

if __name__ == "__main__":
    # Paths
    csv = r"C:\\Users\\123ti\\Documents\\Speciale_git\\Speciale\\Hurricane_data\\IBTrACS_filtered_data.csv"
    genesis = load_genesis_points(csv)
    print(genesis.head())
    # Show plot with points
    #plot_regions_with_genesis(regions, genesis, out_path=out_path)
    # HadISST processing
    ds = xr.open_dataset(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/HadISST_sst.nc")
    sst_hadisst = ds['sst']
    process_sst_dataset(
        sst_hadisst, 
        regions,
        dataset_name="HadISST",
        sample_time="1920-08",
        output_csv=r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/mean_sst_regions_HadISST.csv",
        suffix='hadisst',
        lon_name='longitude',
        lat_name='latitude',
        x_plot='longitude',
        y_plot='latitude'
    )
    
    # ICOADS processing
    icoads_df = xr.open_dataset(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/sst_monthly_mean_icoads.nc")
    icoads_df = icoads_df.assign_coords(lon=((icoads_df.lon + 180) % 360) - 180)
    icoads_df = icoads_df.sortby('lon')
    sst_icoads = icoads_df['sst']
    process_sst_dataset(
        sst_icoads,
        regions,
        dataset_name="ICOADS",
        sample_time="1920-08",
        output_csv=r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/mean_sst_regions_Icoads.csv",
        suffix='icoads',
        lon_name='lon',
        lat_name='lat',
        x_plot='lon',
        y_plot='lat'
    )

    #compare mean temps
    # hadisst_df = pd.read_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/mean_sst_regions_HadISST.csv")
    # icoads_df = pd.read_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/mean_sst_regions_Icoads.csv")
    # compare_df = pd.merge(hadisst_df, icoads_df, on=['Year', 'Month'], suffixes=('_hadisst', '_icoads'))
    # compare_df = compare_df[compare_df['Year'] >= 1900]
    # icoads_gc = compare_df[['mean_temp_gc_icoads']].values
    # hadisst_gc = compare_df[['mean_temp_gc_hadisst']].values
    # diff_gc = icoads_gc - hadisst_gc
    # icoads_mdr = compare_df[['mean_temp_mdr_icoads']].values
    # hadisst_mdr = compare_df[['mean_temp_mdr_hadisst']].values
    # diff_mdr = icoads_mdr - hadisst_mdr
    # plt.figure(figsize=(8,5))
    # plt.subplot(2,1,1)
    # start = 200
    # end = 1500
    # year_month = compare_df['Year'] + (compare_df['Month'] - 1)/12
    # yearly_mean_gc_icoads = pd.Series(icoads_gc.flatten()).rolling(window=12, center=True).mean()
    # yearly_mean_gc_hadisst = pd.Series(hadisst_gc.flatten()).rolling(window=12, center=True).mean()
    # yearly_mean_mdr_icoads = pd.Series(icoads_mdr.flatten()).rolling(window=12, center=True).mean()
    # yearly_mean_mdr_hadisst = pd.Series(hadisst_mdr.flatten()).rolling(window=12, center=True).mean()
    # #plt.plot(year_month[start:end], hadisst_gc[start:end], label='HadISST Gulf+Caribbean', color='tab:blue')
    # #plt.plot(year_month[start:end], icoads_gc[start:end], label='ICOADS Gulf+Caribbean', color='tab:orange')
    # #plt.plot(year_month[start:end], diff_gc[start:end], label='Difference (ICOADS - HadISST)', color='tab:red')
    # plt.plot(year_month[start:end], yearly_mean_gc_hadisst[start:end], label='HadISST Gulf+Caribbean', color='tab:blue')
    # plt.plot(year_month[start:end], yearly_mean_gc_icoads[start:end], label='ICOADS Gulf+Caribbean', color='tab:orange')
    # plt.title('Gulf + Caribbean Mean SST Comparison')
    # plt.legend()


    # plt.subplot(2,1,2)
    # #plt.plot(year_month[start:end], hadisst_mdr[start:end], label='HadISST Main Development Region', color='tab:blue')
    # #plt.plot(year_month[start:end], icoads_mdr[start:end], label='ICOADS Main Development Region', color='tab:orange')
    # #plt.plot(year_month[start:end], diff_mdr[start:end], label='Difference (ICOADS - HadISST)', color='tab:red')
    # plt.plot(year_month[start:end], yearly_mean_mdr_hadisst[start:end], label='HadISST Main Development Region', color='tab:blue')
    # plt.plot(year_month[start:end], yearly_mean_mdr_icoads[start:end], label='ICOADS Main Development Region', color='tab:orange')
    # plt.title('Main Development Region Mean SST Comparison')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()



