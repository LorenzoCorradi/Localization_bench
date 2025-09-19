import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.transform import rowcol
import torch
import pyproj
import csv
import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent / "CAMP"))
from camp_inference import inference_CAMP
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd

# # Parameters
# DRONE_RESOLUTION = 0.094   # m/px
# MAP_RESOLUTION = 0.27      # m/px
# DRONE_SIZE = (3976, 2652)  # px (width, height)
# MAP_SIZE = (9774, 26762)   # px (width, height)
# DRONE_FOLDER = "UAV-VisLoc/01/drone"
# DRONE_CSV = "UAV-VisLoc/01/01.csv"
# SATELLITE_IMG = "UAV-VisLoc/01/satellite01.tif"
# TOP_K = 10
# OVERLAP = 0.2
# SAVE_RETRIEVAL_RESULTS = False
# RESULTS_FOLDER = "retrieval_results"
# ROTATION_ANGLE = 0
# RANSAC_INLIERS = 0.4
# LIGHT_GLUE_MINUMUM_SCORE = 0.95
# SHOW_LG_MATCHES = False
# SHOW_CENTER_PREDICTION = False
# RECALL_K = [1, 2, 5, 10]
# LOCALIZATION_THRESHOLDS = [5, 10, 20, 50]
# RESULTS_CSV = "results_overlap0_top10_rotation0.csv"

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def tensor_to_uint8(img_tensor):
    """Converti un torch.Tensor CxHxW o HxWxC in numpy uint8"""
    if isinstance(img_tensor, torch.Tensor):
        # porta il tensore sulla CPU
        img_np = img_tensor.detach().cpu().numpy()
        # Se CxHxW -> HxWxC
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1,2,0))
        # normalizza [0,1] -> [0,255] se float
        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        return img_np
    return img_tensor


def tensor_to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t

def draw_matches(img0, img1, kpts0, kpts1, matches, max_size=800):
    img0 = tensor_to_uint8(img0)
    img1 = tensor_to_uint8(img1)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    scale0 = max_size / max(h0, w0)
    scale1 = max_size / max(h1, w1)
    scale = min(scale0, scale1, 1.0)

    new_h0, new_w0 = int(h0 * scale), int(w0 * scale)
    new_h1, new_w1 = int(h1 * scale), int(w1 * scale)

    img0 = cv2.resize(img0, (new_w0, new_h0))
    img1 = cv2.resize(img1, (new_w1, new_h1))
    kpts0 = (kpts0.cpu().numpy() * scale).astype(int)
    kpts1 = (kpts1.cpu().numpy() * scale).astype(int)

    h = max(new_h0, new_h1)
    vis = np.zeros((h, new_w0 + new_w1, 3), dtype=np.uint8)
    vis[:new_h0, :new_w0] = img0
    vis[:new_h1, new_w0:new_w0 + new_w1] = img1

    for (idx0, idx1) in matches:
        pt0 = tuple(kpts0[idx0])
        pt1 = tuple(kpts1[idx1] + np.array([new_w0, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(vis, pt0, 3, color, -1)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.line(vis, pt0, pt1, color, 1)

    return vis

def resize_for_display(img, max_width=600):
    h, w = img.shape[:2]
    scale = max_width / w if w > max_width else 1.0
    return cv2.resize(img, (int(w*scale), int(h*scale)))

def rotate_image(img, angle):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    center = (cx, cy)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0,2] += (new_w / 2) - cx
    M[1,2] += (new_h / 2) - cy

    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    M_full = np.vstack([M, [0, 0, 1]])

    return rotated, M_full


def generate_tiles(map_path, tile_size=384, overlap=0.2):
    """Divide the satellite image into tiles with overlap."""
    img = cv2.imread(map_path)
    img_h, img_w = img.shape[:2]

    step_x = int(tile_size * (1 - overlap))
    step_y = int(tile_size * (1 - overlap))

    tiles = []
    positions = []
    tiles_dict = {}

    for y in range(0, img_h - tile_size + 1, step_y):
        for x in range(0, img_w - tile_size + 1, step_x):
            tile = img[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((x + tile_size/2, y + tile_size/2))

    return tiles, positions, (img_w, img_h)

class Tile:
    def __init__(self, image, center, index, tile_size, embedding=None):
        self.image = image
        self.center = center      # (x, y)
        self.index = index        # (i, j) inside the matrix
        self.tile_size = tile_size
        self.embedding = embedding  # può essere None all'inizio

    @property
    def bbox(self):
        """Bounding box (x0, y0, x1, y1) in pixel"""
        cx, cy = self.center
        half = self.tile_size // 2
        return (cx - half, cy - half, cx + half, cy + half)

    def contains(self, px, py):
        """True se il punto (px,py) cade dentro il tile"""
        x0, y0, x1, y1 = self.bbox
        return x0 <= px < x1 and y0 <= py < y1



def generate_tile_matrix(map_path, tile_size=384, overlap=0.2):
    """Divide l'immagine in una matrice NumPy di oggetti Tile."""
    img = cv2.imread(map_path)
    img_h, img_w = img.shape[:2]

    step_x = int(tile_size * (1 - overlap))
    step_y = int(tile_size * (1 - overlap))

    n_rows = (img_h - tile_size) // step_y + 1
    n_cols = (img_w - tile_size) // step_x + 1

    tiles = np.empty((n_rows, n_cols), dtype=object)

    for i, y in enumerate(range(0, img_h - tile_size + 1, step_y)):
        for j, x in enumerate(range(0, img_w - tile_size + 1, step_x)):
            tile_img = img[y:y+tile_size, x:x+tile_size]
            center = (x + tile_size/2, y + tile_size/2)
            tiles[i, j] = Tile(tile_img, center, (i, j), tile_size)

    return tiles, (img_w, img_h), (step_x, step_y)


def get_nearest_tile(px, py, tiles, steps):
    """Trova la tile che contiene (px,py), scegliendo quella col centro più vicino."""
    step_x, step_y = steps
    n_rows, n_cols = tiles.shape

    # raw index
    j0 = px // step_x
    i0 = py // step_y

    candidate_tiles = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            i, j = int(i0 + di), int(j0 + dj)
            if 0 <= i < n_rows and 0 <= j < n_cols:
                tile = tiles[i, j]
                if tile.contains(px, py):
                    candidate_tiles.append(tile)

    if not candidate_tiles:
        return None

    # tile with the nearest center
    best_tile = min(candidate_tiles,
                    key=lambda t: (px - t.center[0])**2 + (py - t.center[1])**2)

    return best_tile


def show_tile(tile, window_name="Nearest Tile", max_screen=(1200, 800)):
    """
    Mostra una singola tile ridimensionandola se troppo grande per lo schermo.
    tile: oggetto Tile
    max_screen: (width, height) massima finestra
    """
    img = tile.image.copy()
    h, w = img.shape[:2]

    # calcola il fattore di scala
    scale = min(max_screen[0] / w, max_screen[1] / h, 1.0)

    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)



def show_nearby_tiles(nearby_tiles, tile_size, px_x=None, px_y=None, max_screen=(1200, 800)):
    """
    Mostra tutte le nearby tiles come un'unica immagine, considerando overlap.
    nearby_tiles: lista di Tile
    tile_size: dimensione del singolo tile
    px_x, px_y: coordinate del punto centrale da evidenziare in rosso
    max_screen: dimensione massima finestra (width, height)
    """
    # Calcola bounding box complessiva (min_x0, min_y0, max_x1, max_y1)
    x0s = [int(t.center[0] - tile_size/2) for t in nearby_tiles]
    y0s = [int(t.center[1] - tile_size/2) for t in nearby_tiles]
    x1s = [int(t.center[0] + tile_size/2) for t in nearby_tiles]
    y1s = [int(t.center[1] + tile_size/2) for t in nearby_tiles]

    min_x0, min_y0 = min(x0s), min(y0s)
    max_x1, max_y1 = max(x1s), max(y1s)

    width = max_x1 - min_x0
    height = max_y1 - min_y0

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Inserisci ogni tile
    for t in nearby_tiles:
        cx, cy = t.center
        x0 = int(cx - tile_size/2 - min_x0)
        y0 = int(cy - tile_size/2 - min_y0)
        x1 = x0 + tile_size
        y1 = y0 + tile_size

        canvas[y0:y1, x0:x1] = t.image

    # Disegna il punto centrale in rosso
    if px_x is not None and px_y is not None:
        canvas_x = int(px_x - min_x0)
        canvas_y = int(px_y - min_y0)
        cv2.circle(canvas, (canvas_x, canvas_y), radius=10, color=(0, 0, 255), thickness=-1)

    # Ridimensiona se troppo grande
    screen_w, screen_h = max_screen
    scale = min(screen_w / width, screen_h / height, 1.0)
    if scale < 1.0:
        canvas = cv2.resize(canvas, (int(width*scale), int(height*scale)))

    cv2.imshow("Nearby Tiles", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_nearby_tiles(px_x, px_y, RANGE, MAP_RESOLUTION, tiles, steps):
    """
    Restituisce tutte le tile entro RANGE metri dal punto px_x, px_y
    """
    step_x, step_y = steps
    tile_radius_x = int((RANGE / MAP_RESOLUTION) / step_x) + 1
    tile_radius_y = int((RANGE / MAP_RESOLUTION) / step_y) + 1

    n_rows, n_cols = tiles.shape

    # tile centrale
    i_center, j_center = get_nearest_tile(px_x, px_y, tiles, steps).index

    nearby_tiles = []
    for di in range(-tile_radius_y, tile_radius_y + 1):
        for dj in range(-tile_radius_x, tile_radius_x + 1):
            i = i_center + di
            j = j_center + dj
            if 0 <= i < n_rows and 0 <= j < n_cols:
                nearby_tiles.append(tiles[i, j])

    if SHOW_NEARBY_TILES:
        show_nearby_tiles(nearby_tiles, tile_size=get_nearest_tile(px_x, px_y, tiles, steps).tile_size, px_x=px_x, px_y=px_y)

    return nearby_tiles



def superpoint_lightglue(extractor, matcher,  tile, drone):
    feats0 = extractor.extract(numpy_image_to_torch(tile).to(device))
    feats1 = extractor.extract(numpy_image_to_torch(drone).to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

    matching_scores0 = matches01["matching_scores0"]
    matching_scores1 = matches01["matching_scores1"]
    matches= matches[matching_scores0[matches[:,0]] > LIGHT_GLUE_MINUMUM_SCORE]

    if matches.shape[0] <10:
        return None, None, None
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    pts0 = m_kpts0.detach().cpu().numpy().astype(np.float32)
    pts1 = m_kpts1.detach().cpu().numpy().astype(np.float32)

    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransacReprojThreshold=3.0)

    inlier_matches = matches[mask.ravel() == 1]
    inlier_kpts0 = m_kpts0[mask.ravel() == 1]
    inlier_kpts1 = m_kpts1[mask.ravel() == 1]

    print(f"{len(inlier_matches)} inliers su {len(matches)} match totali")

    if SHOW_LG_MATCHES:
        vis_inliers = draw_matches(numpy_image_to_torch(tile).to(device), numpy_image_to_torch(drone).to(device), inlier_kpts0, inlier_kpts1, np.array([[i,i] for i in range(len(inlier_matches))]))
        vis_inliers = resize_for_display(vis_inliers, max_width=1200)
        cv2.imshow("Inlier Matches", vis_inliers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return H, len(inlier_matches), len(matches)


if __name__ == "__main__":

    # PARSE CONFIGURATION
    parser = argparse.ArgumentParser(description="Drone Localization Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (JSON)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    DRONE_RESOLUTION = cfg["DRONE_RESOLUTION"]
    MAP_RESOLUTION = cfg["MAP_RESOLUTION"]
    DRONE_SIZE = cfg["DRONE_SIZE"]
    DRONE_FOLDER = Path(cfg["DRONE_FOLDER"])
    DRONE_CSV = cfg["DRONE_CSV"]
    SATELLITE_IMG = cfg["SATELLITE_IMG"]
    TOP_K = cfg["TOP_K"]
    OVERLAP = cfg["OVERLAP"]
    SAVE_RETRIEVAL_RESULTS = cfg["SAVE_RETRIEVAL_RESULTS"]
    RESULTS_FOLDER = Path(cfg["RESULTS_FOLDER"])
    ROTATION_ANGLE = cfg["ROTATION_ANGLE"]
    RANSAC_INLIERS = cfg["RANSAC_INLIERS"]
    LIGHT_GLUE_MINUMUM_SCORE = cfg["LIGHT_GLUE_MINUMUM_SCORE"]
    SHOW_LG_MATCHES = cfg["SHOW_LG_MATCHES"]
    SHOW_CENTER_PREDICTION = cfg["SHOW_CENTER_PREDICTION"]
    RECALL_K = cfg["RECALL_K"]
    LOCALIZATION_THRESHOLDS = cfg["LOCALIZATION_THRESHOLDS"]
    RESULTS_CSV = cfg["RESULTS_CSV"]
    RANGE = cfg["RANGE"]  # meters
    SHOW_NEARBY_TILES = cfg["SHOW_NEARBY_TILES"]
    ALPHA = cfg["ALPHA"]
    

    print(f"Config loaded from {args.config}")
    print(f"Drone resolution: {DRONE_RESOLUTION}, Top-K: {TOP_K}")


    # MAIN ALGORITHM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DRONE_CSV)
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)    


    with rasterio.open(SATELLITE_IMG) as src:
        res_x, res_y = src.res
        crs = src.crs
        bounds = src.bounds
        center_lat = (bounds.top + bounds.bottom) / 2

    if crs.to_epsg() == 4326:
        geod = pyproj.Geod(ellps="WGS84")
        lon_dist = geod.line_length([0, res_x], [center_lat, center_lat])
        lat_dist = geod.line_length([0, 0], [0, res_y])
        MAP_RESOLUTION = min(lon_dist, lat_dist)
        print(MAP_RESOLUTION)
    else:
        print(f"Error")
        exit(0)    

    drone_max_fov_m = max(DRONE_SIZE[0] * DRONE_RESOLUTION, DRONE_SIZE[1] * DRONE_RESOLUTION) * 1.2
    map_tile_size = int(drone_max_fov_m / MAP_RESOLUTION)
    print(f"Tile size: {map_tile_size} x {map_tile_size} px")
    tiles, size, steps = generate_tile_matrix(SATELLITE_IMG, tile_size=map_tile_size, overlap=OVERLAP)

    print(f"Generated {len(tiles)} tiles")

    # GENERATE VECTORS FROM SATELLITE
    batch_size = tiles.size
    tiles_flat = tiles.ravel()  
    images = [t.image for t in tiles_flat]
    embs = inference_CAMP(images)
    for tile, emb in zip(tiles_flat, embs):
        tile.embedding = emb

    
    tiles_features = [tile.embedding for tile in tiles.ravel()]
    with rasterio.open(SATELLITE_IMG) as src:
        transform = src.transform

    count = -1
    all_matches = []
    results = []

    for _, row in df.iterrows():
        # count+=1
        # if count %100 != 0:
        #     continue
        
        filename = row["filename"]
        lat, lon = row["lat"], row["lon"]
        phi = row["Phi1"]

        drone_path = Path(DRONE_FOLDER) / filename
        if not drone_path.exists():
            print(f"Warning: {drone_path} not found, skipping")
            continue

        drone_img = cv2.imread(str(drone_path))
        
        # GENERATE VECTOR FROM DRONE
        query_feature = inference_CAMP([drone_img])[0]

        nearby_tiles = []
        with rasterio.open(SATELLITE_IMG) as src:
            # se il raster non è EPSG:4326, converti le coordinate
            if src.crs.to_epsg() != 4326:
                import pyproj
                transformer = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                lon, lat = transformer.transform(lon, lat)
            
            row, col = rowcol(src.transform, lon, lat)
            px_x, px_y = col, row  
        nearby_tiles = get_nearby_tiles(px_x, px_y, RANGE, MAP_RESOLUTION, tiles, steps)


        nearby_features = [t.embedding for t in nearby_tiles]  # lista di array

        # converte in matrice (N x D)
        X = np.stack(nearby_features)  # shape (N, D)
        q = query_feature.flatten()    # shape (D,)

        X_torch = torch.from_numpy(X)          # shape (N, D)
        q_torch = q.flatten()                  # già tensor
        scores = torch.matmul(X_torch, q_torch)

        top_scores, top_indices = torch.topk(scores, TOP_K)
        top_tiles = [nearby_tiles[i] for i in top_indices.tolist()]

        print(f"Drone image: {filename}, Top {TOP_K} tiles: {top_indices}")

        if SAVE_RETRIEVAL_RESULTS:
            query_folder = Path(RESULTS_FOLDER) / Path(filename).stem
            query_folder.mkdir(parents=True, exist_ok=True)

            scores_file = query_folder / "scores.txt"
            with open(scores_file, "w") as f:
                f.write(f"Drone image: {filename}\n")
                f.write("Top-K scores:\n")
                for rank, idx in enumerate(top_indices):
                    tile = tiles[idx]
                    score = scores[idx]
                    save_path = query_folder / f"top{rank+1}_tile_{idx:04d}.png"
                    cv2.imwrite(str(save_path), tile)
                    f.write(f"  Rank {rank+1}: tile_{idx:04d}, score={score:.4f}\n")

            row_pix, col_pix = rowcol(transform, lon, lat)  
            px, py = col_pix, row_pix

            gt_idx = None
            for i, (x, y) in enumerate(positions):
                if x <= px < x + (1 - OVERLAP) * map_tile_size and y <= py < y + (1 - OVERLAP) * map_tile_size:
                    gt_idx = i
                    break

            if gt_idx is not None:
                gt_tile = tiles[gt_idx]
                cv2.imwrite(str(query_folder / f"GT_tile_{gt_idx:04d}.png"), gt_tile)

                gt_score = float(np.dot(query_feature.flatten(), tiles_features[gt_idx].flatten()))
                with open(scores_file, "a") as f:
                    f.write(f"\nGT tile: tile_{gt_idx:04d}, score={gt_score:.4f}\n")
            else:
                print(f"GT tile not found for {filename} (px={px}, py={py})")


        # COMPUTE_RECALL:
        found_matches_per_query = []
        geod = pyproj.Geod(ellps='WGS84')
        
        for t in top_tiles:
            tile_center_x, tile_center_y = t.center
            tile_lon, tile_lat = rasterio.transform.xy(transform, int(tile_center_y), int(tile_center_x), offset='center')

            _, _, dx = geod.inv(lon, lat, tile_lon, lat)
            _, _, dy = geod.inv(lon, lat, lon, tile_lat)

            is_match = abs(dx) <= DRONE_SIZE[0]* DRONE_RESOLUTION and abs(dy) <= DRONE_SIZE[1] * DRONE_RESOLUTION
            found_matches_per_query.append(int(is_match))
            
        print("matches:")
        print(found_matches_per_query)
        all_matches.append(found_matches_per_query)


        # TRY TO MATCH EACH IMAGE
        if ROTATION_ANGLE==0:
            angles = [0]
        else:
            angles = [ROTATION_ANGLE*i for i in range(int(360/ROTATION_ANGLE))]
        pred_x = None
        pred_y = None

        candidates = []
        scores = []

        for i, tile in enumerate(top_tiles):
            match = False
            satellite_tile = tile
            error = None


            # optimization
            if not  found_matches_per_query[i]:
                continue

            for angle in angles:
                rotated_drone, M_rot = rotate_image(drone_img, angle)       
                H, inliers, matches = superpoint_lightglue(extractor, matcher, satellite_tile.image, rotated_drone)

                if H is not None:
                    scores.append(ALPHA * inliers/matches +  (1- ALPHA) * inliers)
                    candidates.append((rotated_drone, M_rot, H, tile))

        if len(candidates) > 0:
            print("Found homografy")
            print(scores)
            best = np.argmax(scores)
            rotated_drone, M_rot, H,  tile= candidates[best]

            satellite_tile = tile.image

            h_drone, w_drone = rotated_drone.shape[:2]
            H_inv = np.linalg.inv(H) 
            h_drone, w_drone = drone_img.shape[:2]
            cx, cy = w_drone / 2, h_drone / 2

            center_pt = np.array([[[cx, cy]]], dtype=np.float32)
            center_rot = cv2.transform(center_pt, M_rot)[0,0]
            center_warp = cv2.perspectiveTransform(np.array([[[center_rot[0], center_rot[1]]]], dtype=np.float32), H_inv)[0,0]

            tile_x, tile_y = tile.center  
            tile_offset_x = tile_x - map_tile_size / 2
            tile_offset_y = tile_y - map_tile_size / 2

            pred_col_global = center_warp[0] + tile_offset_x
            pred_row_global = center_warp[1] + tile_offset_y

            pred_x, pred_y = rasterio.transform.xy(transform, int(pred_row_global), int(pred_col_global), offset='center')

            geod = pyproj.Geod(ellps='WGS84')
            _, _, error = geod.inv(lon, lat, pred_x, pred_y)

            print("Coordinate prediction: ", pred_x, pred_y)
            print(f"Google Maps (Pred): https://www.google.com/maps/search/?api=1&query={pred_y},{pred_x}")
            print("Coordinate gt: ", lon, lat)
            print(f"Google Maps (GT): https://www.google.com/maps/search/?api=1&query={lat},{lon}")
            print(f"Error: {error:.2f} m")


            if SHOW_CENTER_PREDICTION:
                warped_drone = cv2.warpPerspective(rotated_drone, H_inv, (satellite_tile.shape[:2][1], satellite_tile.shape[:2][0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                px, py = cx + 800, cy + 800   
                point_pt = np.array([[px, py]], dtype=np.float32).reshape(-1,1,2)
                point_rot = cv2.transform(point_pt, M_rot)[0,0]
                point_warp = cv2.perspectiveTransform(np.array([[[point_rot[0], point_rot[1]]]], dtype=np.float32), H_inv)[0,0]
                
                drone_orig_disp = drone_img.copy()
                drone_rot_disp = rotated_drone.copy()
                warp_disp = warped_drone.copy()
                tile_disp = satellite_tile.copy()

                cv2.circle(drone_orig_disp, (int(cx), int(cy)), 10, (0,0,255), -1)
                cv2.circle(drone_rot_disp, (int(center_rot[0]), int(center_rot[1])), 10, (0,0,255), -1)
                cv2.circle(warp_disp, (int(center_warp[0]), int(center_warp[1])), 10, (0,0,255), -1)
                cv2.circle(tile_disp, (int(center_warp[0]), int(center_warp[1])), 10, (0,0,255), -1)

                cv2.circle(drone_orig_disp, (int(px), int(py)), 10, (0,255,0), -1)
                cv2.circle(drone_rot_disp, (int(point_rot[0]), int(point_rot[1])), 10, (0,255,0), -1)
                cv2.circle(warp_disp, (int(point_warp[0]), int(point_warp[1])), 10, (0,255,0), -1)
                cv2.circle(tile_disp, (int(point_warp[0]), int(point_warp[1])), 10, (0,255,0), -1)

                def resize_for_view(img, max_height=400):
                    h, w = img.shape[:2]
                    if h > max_height:
                        scale = max_height / h
                        return cv2.resize(img, (int(w*scale), int(h*scale)))
                    return img

                drone_orig_disp = resize_for_view(drone_orig_disp)
                drone_rot_disp = resize_for_view(drone_rot_disp)
                warp_disp = resize_for_view(warp_disp)
                tile_disp = resize_for_view(tile_disp)

                vis = np.hstack([drone_orig_disp, drone_rot_disp, warp_disp, tile_disp])
                cv2.imshow("Original | Rotated | Warped | Tile", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        results.append({
            "filename": filename,
            "gt_lon": lon,
            "gt_lat": lat,
            "pred_lon": pred_x if pred_x is not None else "",
            "pred_lat": pred_y if pred_y is not None else "",
            "error_m": error if error is not None else ""
        })

    
    # COMPUTE_RECALL:
    all_matches = np.array(all_matches) 

    for K in RECALL_K:
        R_at_K = (all_matches[:, :K].sum(axis=1) > 0).mean()
        print(f"R@{K}: {R_at_K:.3f}")    


    # COMPUTE ERRORS
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print("Saved results to " + RESULTS_CSV)

    results_df = pd.read_csv(RESULTS_CSV)
    print("\nLocalization Accuracy Results (A@T):")
    for T in LOCALIZATION_THRESHOLDS:
        N = len(results_df)
        if N == 0:
            print(f"A@{T}m: N/A (No valid prediction)")
            continue
        NT = (results_df["error_m"] <= T).sum()
        A_at_T = NT / N * 100
        print(f"A@{T}m: {A_at_T:.2f}%  ({NT}/{N})")


