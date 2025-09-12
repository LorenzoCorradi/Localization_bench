import cv2
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pathlib import Path

# --- Config ---
DRONE_CSV = "UAV-VisLoc/01/01.csv"           # CSV con colonne: filename, lat, lon
SATELLITE_IMG = "UAV-VisLoc/01/satellite01.tif"
OUTPUT_IMG = "satellite_with_gt_path.png"

# --- Load CSV ---
df = pd.read_csv(DRONE_CSV)

# --- Open satellite image ---
with rasterio.open(SATELLITE_IMG) as src:
    img = src.read([1,2,3])                   # assume RGB tiff
    img = img.transpose(1,2,0)                # da CxHxW a HxWxC
    img_copy = img.copy()
    transform = src.transform

# --- Convert all GPS coordinates to pixel coordinates ---
pixels = []
for _, row in df.iterrows():
    lat, lon = row["lat"], row["lon"]
    row_pix, col_pix = rowcol(transform, lon, lat)
    pixels.append((col_pix, row_pix))

# --- Draw lines connecting consecutive points ---
for i in range(1, len(pixels)):
    cv2.line(img_copy, pixels[i-1], pixels[i], color=(0,0,255), thickness=10)  # blu

# --- Optionally draw points ---
for pt in pixels:
    cv2.circle(img_copy, pt, radius=15, color=(255,0,0), thickness=-1)  # rosso

# --- Save result ---
cv2.imwrite(OUTPUT_IMG, img_copy)
print(f"Saved satellite image with GT path to {OUTPUT_IMG}")
