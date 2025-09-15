import os
import time
import math
import shutil
import sys
import torch
import argparse
import numpy as np

from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup

from sample4geo.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
from sample4geo.hand_convnext.model import make_model

import os
import shutil
import numpy as np
import cv2
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class Configuration:
    def __init__(self):
        # Added for your modification
        self.model = 'convnext_base.fb_in22k_ft_in1k_384'
        self.handcraft_model = True
        self.img_size = 384
        self.views = 2
        self.record = True

        self.only_test = True
        self.ckpt_path = './CAMP/weight_mine.pth'

        # Model Config
        self.nclasses = 200
        self.block = 2
        self.triplet_loss = 0.3
        self.resnet = False

        # Our tricks
        self.weight_D_S = 1.0
        self.weight_D_D = 0.
        self.weight_S_S = 0.
        self.weight_D_fine_S_fine = 1.0
        self.weight_D_fine_D_fine = 0.
        self.weight_S_fine_S_fine = 0.

        self.if_learn_ECE_weights = True
        self.learn_weight_D_D = 0.
        self.learn_weight_S_S = 0.
        self.learn_weight_D_fine_S_fine = 0.
        self.learn_weight_D_fine_D_fine = 0.5
        self.learn_weight_S_fine_S_fine = 0.

        # =========================================================================
        self.blocks_for_PPB = 3
        
        self.if_use_plus_1 = False
        self.if_use_multiply_1 = True
        self.only_DS = False
        self.only_fine = True
        self.DS_and_fine = False

        # Original setting
        self.weight_infonce = 1.0
        self.weight_triplet = 0.
        self.weight_cls = 0.
        self.weight_fine = 0.
        self.weight_channels = 0.       # 0.2
        self.weight_dsa = 0.
        self.pos_scale = 0.1            # 0.6
        self.infoNCE_logit = 3.65

        # Training Config
        self.mixed_precision = True
        self.custom_sampling = True
        self.seed = 1
        self.epochs = 1
        self.batch_size = 24
        self.verbose = True
        self.gpu_ids = (0, 1, 2, 3)

        # Eval Config
        self.batch_size_eval = 128
        self.eval_every_n_epoch = 1
        self.normalize_features = True
        self.eval_gallery_n = -1

        # Optimizer Config
        self.clip_grad = 100.0
        self.decay_exclue_bias = False
        self.grad_checkpointing = False

        # Loss Config
        self.label_smoothing = 0.1

        # Learning Rate Config
        self.lr = 0.001   # 1e-4 for ViT | 1e-1 for CNN
        self.scheduler = "cosine"   # "polynomial" | "cosine" | "constant" | None
        self.warmup_epochs = 0.1
        self.lr_end = 0.0001

        # Learning part Config
        self.lr_mlp = None
        self.lr_decouple = None
        self.lr_blockweights = 2
        self.lr_weight_ECE = None

        # Dataset Config
        self.dataset = 'U1652-D2S'   # 'U1652-D2S' | 'U1652-S2D'
        self.altitude = 300          # 150|200|250|300|666, 666 is all data
        self.data_folder = './data/test_dataset_estremo'
        self.dataset_name = 'test_dataset_estremo'

        # Augment Images Config
        self.prob_flip = 0.5

        # Savepath for model checkpoints Config
        self.model_path = './checkpoints/sues-200'

        # Eval before training Config
        self.zero_shot = False

        # Checkpoint to start from Config
        self.checkpoint_start = None

        # Set num_workers to 0 if on Windows
        self.num_workers = 0 if os.name == 'nt' else 4

        # Train on GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # For better performance
        self.cudnn_benchmark = True

        # Make cudnn deterministic
        self.cudnn_deterministic = True


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#


def preprocess_images(images, val_transforms):
    processed = []
    for img in images:
        
        if img.shape[2] == 3:  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = val_transforms(image=img)['image']  
        processed.append(transformed)
    return torch.stack(processed)  

def batch_inference(model, images, val_transforms, device='cuda', batch_size=8):
    model.eval()
    all_results = []

    images_tensor = preprocess_images(images, val_transforms)
    images_tensor = images_tensor.to(device)

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Batch Inference"):
            batch = images_tensor[i:i+batch_size]
            result = model(batch)       
            features = result[0]        
            features = F.normalize(features, dim=-1)
            all_results.extend(features.cpu())

    return all_results


def inference_CAMP(images):
    config = Configuration()
    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    


    model = make_model(config)

    # -- print weight config infos   weight_channels
    print(f"\nweight_infonce:{config.weight_infonce}\nweight_fine:{config.weight_fine}\n"
          f"infoNCE_logit:{config.infoNCE_logit}\npos_scale:{config.pos_scale}\n"
          f"weight_channels:{config.weight_channels}\n")

    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)


    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std))

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    # -----------------------------------------------------------------------------#
    # Test Only                                                                    #
    # -----------------------------------------------------------------------------#

    checkpoint = torch.load(config.ckpt_path)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    results = batch_inference(model, images, val_transforms, config.device, 8)
    return results