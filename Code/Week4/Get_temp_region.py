import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import pandas as pd
import numpy as np

# Define bounding boxes using -180..180 longitudes
regions = {
    # Gulf of Mexico + Caribbean Sea (approx)
    "Gulf + Caribbean": {
        "lon_min": -98.0,
        "lon_max": -60.0,
        "lat_min": 8.0,
        "lat_max": 33.0,
        "color": "tab:blue",
    },
    # Atlantic Main Development Region (MDR) for hurricanes (approx)
    "Atlantic MDR": {
        "lon_min": -60.0,
        "lon_max": -15.0,
        "lat_min": 9.0,
        "lat_max": 25.0,
        "color": "tab:red",
    },
}


def plot_regions(regions_dict, out_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Compute combined extent with margin
    lons_min = [r["lon_min"] for r in regions_dict.values()]
    lons_max = [r["lon_max"] for r in regions_dict.values()]
    lats_min = [r["lat_min"] for r in regions_dict.values()]
    lats_max = [r["lat_max"] for r in regions_dict.values()]
    lon_min = min(lons_min) - 5
    lon_max = max(lons_max) + 5
    lat_min = min(lats_min) - 5
    lat_max = max(lats_max) + 5
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Map features
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.3)
    gl.right_labels = False
    gl.top_labels = False

    # Draw each region
    for name, r in regions_dict.items():
        rect = mpatches.Rectangle(
            (r["lon_min"], r["lat_min"]),
            r["lon_max"] - r["lon_min"],
            r["lat_max"] - r["lat_min"],
            linewidth=1.6,
            edgecolor=r.get("color", "black"),
            facecolor="none",
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(rect)

        # Label near top-left corner of the box
        ax.text(
            r["lon_min"] + 0.5,
            r["lat_max"] - 0.5,
            name,
            fontsize=10,
            color=r.get("color", "black"),
            transform=ccrs.PlateCarree(),
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
        )

    ax.set_title("Hurricane Regions: Gulf+Caribbean and Atlantic MDR")

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


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

    # Combined extent
    lons_min = [r["lon_min"] for r in regions_dict.values()]
    lons_max = [r["lon_max"] for r in regions_dict.values()]
    lats_min = [r["lat_min"] for r in regions_dict.values()]
    lats_max = [r["lat_max"] for r in regions_dict.values()]
    lon_min = min(lons_min) - 1
    lon_max = max(lons_max) + 3
    lat_min = min(lats_min) - 3
    lat_max = max(lats_max) + 3
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Basemap
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.3)
    gl.right_labels = False
    gl.top_labels = False

    # Regions
    for name, r in regions_dict.items():
        rect = mpatches.Rectangle(
            (r["lon_min"], r["lat_min"]),
            r["lon_max"] - r["lon_min"],
            r["lat_max"] - r["lat_min"],
            linewidth=1.6,
            edgecolor=r.get("color", "black"),
            facecolor="none",
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(rect)
        ax.text(
            r["lon_min"] + 0.5,
            r["lat_max"] - 0.5,
            name,
            fontsize=10,
            color=r.get("color", "black"),
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

    ax.set_title("Genesis positions with Gulf/Caribbean + Atlantic MDR boxes")
    ax.legend(loc="lower left")

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    # Paths
    csv = r"C:\\Users\\123ti\\Documents\\Speciale_git\\Speciale\\Hurricane_data\\IBTrACS_filtered_data.csv"
    genesis = load_genesis_points(csv)
    # Show plot with points
    out_path = r"C:\\Users\\123ti\\Documents\\Speciale_git\\Speciale\\Week4\\Important_plots\\Hurricane_regions_with_genesis.png"
    plot_regions_with_genesis(regions, genesis, out_path=out_path )
    hurricane_data = pd.read_csv(r"./Speciale/Hurricane_data/Aslak_data_with_tide.csv")
    relevant_data = hurricane_data[['ATCF_ID', 'lf_ISO_TIME', 'lf_wind', 'lf_pressure' , 'lf_lat', 'lf_lon']]
    Ibtracs_data = pd.read_csv(r"./Speciale/Hurricane_data/IBTrACS_filtered_data.csv")
    hurricane_count = len(relevant_data['ATCF_ID'])
    print(f"Total hurricanes in dataset: {hurricane_count}")
    i=0
    for index, row in relevant_data.iterrows():

        hurricane_id = row['ATCF_ID']
        filtered_IBTrACS_data = Ibtracs_data[Ibtracs_data['USA_ATCF_ID'] == hurricane_id]
        initial_lat = np.array(filtered_IBTrACS_data['LAT'])[0]
        initial_lon = np.array(filtered_IBTrACS_data['LON'])[0]
        position = (initial_lon, initial_lat)
        if i== 0:
            print("Checking hurricane genesis positions against defined regions...")
            print("Hurricane ID | Genesis Position (Lon, Lat) | Region Check")
            print(f"{hurricane_id} | {position} | ", end="")
        if position[0] < -90 and position[1] < 18: # Convert to -180..180
            print(f"{hurricane_id} | {position} | Gulf + Caribbean")
            
        #check if hurricane started in one of the regions
        i_before = i
        for region_name, region in regions.items():
            if (region['lon_min'] <= initial_lon <= region['lon_max'] and
                region['lat_min'] <= initial_lat <= region['lat_max']):
                i += 1
        if i == i_before:
            continue
            print(hurricane_id, "started in region at position", position)



