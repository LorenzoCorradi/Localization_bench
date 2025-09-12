import time
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

torch.set_grad_enabled(False)
images = Path("assets")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
#extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(features="superpoint").eval().to(device)
image0 = load_image(images / "prova1.jpg")
image1 = load_image(images / "prova2.png")

# Misuro il tempo di estrazione features immagine 0
start_time = time.time()
feats0 = extractor.extract(image0.to(device))
end_time = time.time()
print(f"Feature extraction image0 took {end_time - start_time:.4f} seconds")

# Misuro il tempo di estrazione features immagine 1
start_time = time.time()
feats1 = extractor.extract(image1.to(device))
end_time = time.time()
print(f"Feature extraction image1 took {end_time - start_time:.4f} seconds")

matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
print(m_kpts0)

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
viz2d.save_plot("test2_disk.png")

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
