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
import os

def generate_temp_data(df):
    # Define regions using shapefiles
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




    def process_sst_dataset(sst, regions_dict, dataset_name, sample_time, suffix, 
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
        merged_df.drop(columns=['region_gc', 'region_mdr'], inplace=True)
        
        return merged_df

    #check if file exists?
    if os.path.exists(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/generated_data/mean_sst_regions_HadISST.csv") and os.path.exists(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/generated_data/mean_sst_regions_ICOADS.csv"):
        print("Regional temp data already exists. Loading from file.")
        return pd.read_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/generated_data/mean_sst_regions_HadISST.csv"), pd.read_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing/generated_data/mean_sst_regions_ICOADS.csv")

    # HadISST processing
    ds = xr.open_dataset(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/HadISST_sst.nc")
    sst_hadisst = ds['sst']
    hadisst_frame = process_sst_dataset(
        sst_hadisst, 
        regions,
        dataset_name="HadISST",
        sample_time="1920-08",
        suffix='hadisst',
        lon_name='longitude',
        lat_name='latitude',
        x_plot='longitude',
        y_plot='latitude'
    )
    hadisst_frame.to_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing"
                            "/generated_data/mean_sst_regions_HadISST.csv", index=False)
    
    
    # ICOADS processing
    icoads_df = xr.open_dataset(r"C:/Users/123ti/Documents/Speciale_git/Speciale/temp_data/sst_monthly_mean_icoads.nc")
    icoads_df = icoads_df.assign_coords(lon=((icoads_df.lon + 180) % 360) - 180)
    icoads_df = icoads_df.sortby('lon')
    sst_icoads = icoads_df['sst']
    icoads_frame = process_sst_dataset(
        sst_icoads,
        regions,
        dataset_name="ICOADS",
        sample_time="1920-08",
        suffix='icoads',
        lon_name='lon',
        lat_name='lat',
        x_plot='lon',
        y_plot='lat'
    )
    icoads_frame.to_csv(r"C:/Users/123ti/Documents/Speciale_git/Speciale/Code/Data_preprocessing"
                            "/generated_data/mean_sst_regions_ICOADS.csv", index=False)
    return hadisst_frame, icoads_frame


#for plotting
# valid_IDs = df['ATCF_ID'].unique()
# # Paths
# ibtracks_data = pd.read_csv(r"C:\Users\123ti\Documents\Speciale_git\Speciale\Hurricane_data\ibtracs.ALL.list.v04r01.csv", low_memory=False)
# ibtracks_data_filtered = ibtracks_data[ibtracks_data['USA_ATCF_ID'].isin(valid_IDs)]
# genesis = load_genesis_points(ibtracks_data_filtered)

# Show plot with points
#plot_regions_with_genesis(regions, genesis)


def load_genesis_points(df):

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