import os
from pathlib import Path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from xarray.coders import CFDatetimeCoder  # xarray >= 2024 API

# Input file
nc_path = r"C:\Users\123ti\Downloads\ICOADS_R3.0.0_1900-08.nc"

# Output folder
out_dir = Path("./Speciale/Code/Week4/Plots")
out_dir.mkdir(parents=True, exist_ok=True)

# Bbox (note: here longitudes are 0–360)
bbox = {
    "lon_min": 275.463867,
    "lon_max": 332.578125,
    "lat_min": 10.893547,
    "lat_max": 31.319849,
}

# Load dataset
nc_path = r"C:\Users\123ti\Downloads\ICOADS_R3.0.0_1900-08.nc"

# Preferred: use CFDatetimeCoder with cftime (no deprecation warning)


time_coder = CFDatetimeCoder(use_cftime=True)
ds = xr.open_dataset(nc_path, decode_cf=True, decode_times=time_coder)

print("First 3 decoded times:", ds["time"].values[100:110])
print("Time dtype:", ds["time"].dtype)
# Ensure longitude in 0–360 for masking
if "lon" not in ds:
    raise KeyError("Dataset has no 'lon' coordinate.")
if "lat" not in ds:
    raise KeyError("Dataset has no 'lat' coordinate.")

lon360 = (ds["lon"] % 360)
lat = ds["lat"]

mask = (
    (lon360 >= bbox["lon_min"]) & (lon360 <= bbox["lon_max"]) &
    (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"])
)

# Drop everything outside bbox
ds_sel = ds.where(mask, drop=True)

# Extract and flatten lon/lat
lons = np.asarray((ds_sel["lon"] % 360)).ravel()
lats = np.asarray(ds_sel["lat"]).ravel()

valid = np.isfinite(lons) & np.isfinite(lats)
lons = lons[valid]
lats = lats[valid]

# Convert longitudes to -180..180 for plotting with PlateCarree()
lons_plot = np.where(lons > 180.0, lons - 360.0, lons)

# Optional: pick a variable to color the points (fallback to plain red)
var_name = "SST"
vals = None
if var_name is not None:
    vals_full = np.asarray(ds_sel[var_name]).ravel()
    if vals_full.size == lons.size:
        vals = vals_full
    else:
        # Shape mismatch: just skip coloring
        var_name = None
#remove invalid SST's
if vals is not None:
    #remove values if SST is below 0 deg C
    invalid_sst = vals < 0
    vals = vals[~invalid_sst]
    lons_plot = lons_plot[~invalid_sst]
    lats = lats[~invalid_sst]

# Plot
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

# Convert bbox to -180..180 for extent
lon_min_plot = bbox["lon_min"] - 360 if bbox["lon_min"] > 180 else bbox["lon_min"]
lon_max_plot = bbox["lon_max"] - 360 if bbox["lon_max"] > 180 else bbox["lon_max"]
ax.set_extent([lon_min_plot -5, lon_max_plot +5, bbox["lat_min"] -5, bbox["lat_max"] +5], crs=ccrs.PlateCarree())

# Map features
ax.add_feature(cfeature.LAND, facecolor="0.9")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.3)
gl.right_labels = False
gl.top_labels = False


# Scatter points
if vals is not None:
    sc = ax.scatter(lons_plot, lats, c=vals, s=6, cmap="viridis",
                    transform=ccrs.PlateCarree(), alpha=0.8)
    cb = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(var_name)
else:
    ax.scatter(lons_plot, lats, s=6, color="tab:red",
               transform=ccrs.PlateCarree(), alpha=0.7)

# Draw bbox rectangle
rect = mpatches.Rectangle(
    (lon_min_plot, bbox["lat_min"]),
    lon_max_plot - lon_min_plot,
    bbox["lat_max"] - bbox["lat_min"],
    linewidth=1.0,
    edgecolor="black",
    facecolor="none",
    transform=ccrs.PlateCarree(),
)
ax.add_patch(rect)

ax.set_title(f"ICOADS observations in bbox: {lons.size} points")
out_path = out_dir / "icoads_bbox_observations.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved plot to: {out_path.resolve()}")

