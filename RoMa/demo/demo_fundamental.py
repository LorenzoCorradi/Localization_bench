import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from romatch import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/prova1.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/prova2.png", type=str)
    parser.add_argument("--certainty_thresh", default=0.95, type=float, help="Threshold for showing matches")
    args, _ = parser.parse_known_args()

    im1_path = args.im_A_path
    im2_path = args.im_B_path
    certainty_thresh = args.certainty_thresh

    # Load model
    roma_model = roma_outdoor(device=device)

    # Carica immagini
    im1_pil = Image.open(im1_path).convert("RGB")
    im2_pil = Image.open(im2_path).convert("RGB")

    # Ridimensiona entrambe alla stessa dimensione (usiamo dimensioni di im1)
    W, H = im1_pil.size
    im2_pil = im2_pil.resize((W, H))

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    print(certainty)
    # Sample matches
    matches, certainty_sampled = roma_model.sample(warp, certainty)

    # Converti in coordinate pixel
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H, W, H, W)

    # Filtra solo le corrispondenze con alta certezza
    mask_certainty = certainty_sampled > certainty_thresh
    kpts1_filtered = kpts1[mask_certainty]
    kpts2_filtered = kpts2[mask_certainty]

    # --- Visualization ---
    im1 = np.array(im1_pil)
    im2 = np.array(im2_pil)

    # Canvas affiancato
    canvas = np.zeros((H, 2*W, 3), dtype=np.uint8)
    canvas[:, :W, :] = im1
    canvas[:, W:, :] = im2

import matplotlib.cm as cm

skipped = 0

# Genera una mappa di colori (viridis, tab20, hsv… puoi cambiare)
num_matches = len(kpts1_filtered)
colors = cm.get_cmap('hsv', num_matches)  # 'hsv' dà una ruota di colori

for idx, ((x1, y1), (x2, y2)) in enumerate(zip(kpts1_filtered.cpu().numpy(),
                                               kpts2_filtered.cpu().numpy())):
    skipped += 1
    if skipped == 40:  # disegna 1 linea ogni 40
        skipped = 0
        # Prendi colore in BGR perché OpenCV usa BGR
        color_rgb = colors(idx)[:3]  # (r,g,b) in [0,1]
        color_bgr = tuple(int(c*255) for c in color_rgb[::-1])
        cv2.line(canvas,
                 (int(x1), int(y1)),
                 (int(x2)+W, int(y2)),
                 color=color_bgr,
                 thickness=1)

# Visualizza
plt.figure(figsize=(15,8))
plt.imshow(canvas)
plt.axis('off')
plt.show()
