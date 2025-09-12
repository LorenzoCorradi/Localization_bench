import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from romatch.utils.utils import tensor_to_pil
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
    parser.add_argument("--save_path", default="demo/test.png", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))

    H, W = roma_model.get_output_resolution()

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
    x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
    x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    white_im = torch.ones((H,2*W),device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    tensor_to_pil(vis_im, unnormalize=False).save(save_path)


        # Carica le immagini
    im1 = np.array(Image.open(im1_path).convert("RGB").resize((W,H)))
    im2 = np.array(Image.open(im2_path).convert("RGB").resize((W,H)))

    # warp contiene, per ogni pixel, le coordinate corrispondenti nell'altra immagine
    # Estraggo un campione di pixel per visualizzare le linee
    stride = 20  # prende ogni 20 pixel per non avere troppe linee
    y, x = np.mgrid[0:H:stride, 0:W:stride]

    # Ottieni le coordinate corrispondenti dall'immagine 2
    warp_coords = warp.detach().cpu().numpy()
    pts1 = np.stack([x.ravel(), y.ravel()], axis=1)
    pts2 = warp_coords[y, x, :2].reshape(-1,2)  # prima immagine -> seconda immagine

    # Crea canvas con le due immagini affiancate
    canvas = np.zeros((H, W*2, 3), dtype=np.uint8)
    canvas[:, :W, :] = im1
    canvas[:, W:, :] = im2

    # Trasla le coordinate della seconda immagine
    pts2[:,0] += W

    # Visualizza con matplotlib
    plt.figure(figsize=(15,8))
    plt.imshow(canvas)
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        plt.plot([x1, x2], [y1, y2], 'r', linewidth=0.5)  # linee rosse
    plt.axis('off')
    plt.show()