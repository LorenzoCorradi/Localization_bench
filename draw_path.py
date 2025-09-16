import cv2
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import numpy as np

# --- Config ---
DRONE_CSV = "results_config_8.csv"           # CSV with: filename, gt_lon, gt_lat, pred_lon, pred_lat
SATELLITE_IMG = "UAV-VisLoc/01/satellite01.tif"
OUTPUT_IMG = "satellite_with_gt_and_pred.png"

# --- Load CSV ---
df = pd.read_csv(DRONE_CSV)

# --- Open satellite image ---
with rasterio.open(SATELLITE_IMG) as src:
    # leggo le bande RGB e trasformo in HxWxC
    img = src.read([1, 2, 3]).transpose(1, 2, 0)
    # converto da RGB → BGR per compatibilità con OpenCV
    img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    transform = src.transform

# --- Convert GT and Pred coordinates to pixel coordinates ---
gt_pixels = []
pred_pixels = []

for _, row in df.iterrows():
    # Ground truth
    lat_gt, lon_gt = row["gt_lat"], row["gt_lon"]
    r_gt, c_gt = rowcol(transform, lon_gt, lat_gt)
    gt_pixels.append((c_gt, r_gt))

    # Predizione (potrebbe essere NaN → la salto)
    if not (pd.isna(row["pred_lat"]) or pd.isna(row["pred_lon"])):
        lat_pred, lon_pred = row["pred_lat"], row["pred_lon"]
        r_pred, c_pred = rowcol(transform, lon_pred, lat_pred)
        pred_pixels.append((c_pred, r_pred))
    else:
        pred_pixels.append(None)

# --- Draw GT path (rosso) con numero ---
for i, pt in enumerate(gt_pixels):
    cv2.circle(img_copy, pt, radius=10, color=(0, 0, 255), thickness=-1)
    cv2.putText(img_copy, str(i), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

for i in range(1, len(gt_pixels)):
    cv2.line(img_copy, gt_pixels[i - 1], gt_pixels[i], color=(0, 0, 255), thickness=6)

# --- Draw Prediction path (verde) con numero ---
pred_valid = [p for p in pred_pixels if p is not None]
for i, pt in enumerate(pred_pixels):
    if pt is not None:
        cv2.circle(img_copy, pt, radius=8, color=(0, 255, 0), thickness=-1)
        cv2.putText(img_copy, str(i), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

for i in range(1, len(pred_pixels)):
    if pred_pixels[i - 1] is not None and pred_pixels[i] is not None:
        cv2.line(img_copy, pred_pixels[i - 1], pred_pixels[i], color=(0, 255, 0), thickness=4)


# --- Save result ---
cv2.imwrite(OUTPUT_IMG, img_copy)
print(f"Saved satellite image with GT (red) and prediction (green) to {OUTPUT_IMG}")
