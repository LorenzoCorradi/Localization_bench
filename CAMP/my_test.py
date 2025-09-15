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

from sample4geo.loss.loss import InfoNCE
from sample4geo.loss.triplet_loss import TripletLoss, Tripletloss
from sample4geo.loss.blocks_infoNCE import blocks_InfoNCE
from sample4geo.loss.blocks_infoNCE_channels import blocks_InfoNCE_channels
from sample4geo.loss.square_infoNCE_loss import square_InfoNCE
from sample4geo.loss.blocks_infoNCE_PCA import blocks_InfoNCE_PCA
from sample4geo.loss.blocks_mse import blocks_mse
from sample4geo.loss.peaks_infoNCE import peaks_InfoNCE
from sample4geo.loss.DRO_loss import DRO_Loss
from sample4geo.loss.DSA_loss import DSA_loss

from sample4geo.model import TimmModel
from torch.utils.tensorboard import SummaryWriter
from sample4geo.trainer import predict


@dataclass
class Configuration:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train and Test on SUES-200 dataset')

        # Added for your modification
        parser.add_argument('--model', default='convnext_base.fb_in22k_ft_in1k_384', type=str, help='backbone model')
        parser.add_argument('--handcraft_model', default=True, type=bool, help='use modified backbone')
        parser.add_argument('--img_size', default=384, type=int, help='input image size')
        parser.add_argument('--views', default=2, type=int, help='only supports 2 branches retrieval')
        parser.add_argument('--record', default=True, type=bool, help='use tensorboard to record training procedure')

        parser.add_argument('--only_test', default=True, type=bool, help='use pretrained model to test')
        parser.add_argument('--ckpt_path', default='./checkpoints/sues-200/convnext_base.fb_in22k_ft_in1k_384/143536/weights_e1_0.9540.pth', type=str, help='path to pretrained checkpoint file')

        # Model Config
        parser.add_argument('--nclasses', default=200, type=int, help='sues-200数据集的场景类别数')
        parser.add_argument('--block', default=2, type=int)
        parser.add_argument('--triplet_loss', default=0.3, type=float)
        parser.add_argument('--resnet', default=False, type=bool)

        # Our tricks
        # parser.add_argument('--weight_infonce', default=1.0, type=float)
        # parser.add_argument('--weight_triplet', default=0.2, type=float)
        # parser.add_argument('--weight_cls', default=0.1, type=float)
        # parser.add_argument('--weight_fine', default=0.7, type=float)

        parser.add_argument('--weight_D_S', default=1.0, type=float)
        parser.add_argument('--weight_D_D', default=0., type=float)
        parser.add_argument('--weight_S_S', default=0., type=float)
        parser.add_argument('--weight_D_fine_S_fine', default=1.0, type=float)
        parser.add_argument('--weight_D_fine_D_fine', default=0., type=float)
        parser.add_argument('--weight_S_fine_S_fine', default=0., type=float)

        parser.add_argument('--if_learn_ECE_weights', default=True, type=bool)
        parser.add_argument('--learn_weight_D_D', default=0., type=float)
        parser.add_argument('--learn_weight_S_S', default=0., type=float)
        parser.add_argument('--learn_weight_D_fine_S_fine', default=0., type=float)
        parser.add_argument('--learn_weight_D_fine_D_fine', default=0.5, type=float)
        parser.add_argument('--learn_weight_S_fine_S_fine', default=0., type=float)

        # =========================================================================
        parser.add_argument('--blocks_for_PPB', default=3, type=int)
        
        parser.add_argument('--if_use_plus_1', default=False, type=bool)
        parser.add_argument('--if_use_multiply_1', default=True, type=bool)
        parser.add_argument('--only_DS', default=False, type=bool)
        parser.add_argument('--only_fine', default=True, type=bool)
        parser.add_argument('--DS_and_fine', default=False, type=bool)

        # Original setting
        # Our tricks
        parser.add_argument('--weight_infonce', default=1.0, type=float)
        parser.add_argument('--weight_triplet', default=0., type=float)
        parser.add_argument('--weight_cls', default=0., type=float)
        parser.add_argument('--weight_fine', default=0., type=float)
        parser.add_argument('--weight_channels', default=0., type=float)       # 0.2
        parser.add_argument('--weight_dsa', default=0., type=float)
        parser.add_argument('--pos_scale', default=0.1, type=float)         # 0.6
        parser.add_argument('--infoNCE_logit', default=3.65, type=float)

        # Training Config
        parser.add_argument('--mixed_precision', default=True, type=bool)
        parser.add_argument('--custom_sampling', default=True, type=bool)
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--epochs', default=1, type=int, help='1 epoch for 1652')
        parser.add_argument('--batch_size', default=24, type=int, help='remember the bs is for 2 branches')
        parser.add_argument('--verbose', default=True, type=bool)
        parser.add_argument('--gpu_ids', default=(0, 1, 2, 3), type=tuple)

        # Eval Config
        parser.add_argument('--batch_size_eval', default=128, type=int)
        parser.add_argument('--eval_every_n_epoch', default=1, type=int)
        parser.add_argument('--normalize_features', default=True, type=bool)
        parser.add_argument('--eval_gallery_n', default=-1, type=int)

        # Optimizer Config
        parser.add_argument('--clip_grad', default=100.0, type=float)
        parser.add_argument('--decay_exclue_bias', default=False, type=bool)
        parser.add_argument('--grad_checkpointing', default=False, type=bool)

        # Loss Config
        parser.add_argument('--label_smoothing', default=0.1, type=float)

        # Learning Rate Config
        parser.add_argument('--lr', default=0.001, type=float, help='1 * 10^-4 for ViT | 1 * 10^-1 for CNN')
        parser.add_argument('--scheduler', default="cosine", type=str, help=r'"polynomial" | "cosine" | "constant" | None')
        parser.add_argument('--warmup_epochs', default=0.1, type=float)
        parser.add_argument('--lr_end', default=0.0001, type=float)

        # Learning part Config
        parser.add_argument('--lr_mlp', default=None, type=float)
        parser.add_argument('--lr_decouple', default=None, type=float)
        parser.add_argument('--lr_blockweights', default=2, type=float)
        parser.add_argument('--lr_weight_ECE', default=None, type=float)

        # Dataset Config
        parser.add_argument('--dataset', default='U1652-D2S', type=str, help="'U1652-D2S' | 'U1652-S2D'")
        parser.add_argument('--altitude', default=300, type=int, help="150|200|250|300|666, 666 is all data")
        parser.add_argument('--data_folder', default='./data/test_dataset_estremo', type=str)
        parser.add_argument('--dataset_name', default='test_dataset_estremo', type=str)

        # Augment Images Config
        parser.add_argument('--prob_flip', default=0.5, type=float, help='flipping the sat image and drone image simultaneously')

        # Savepath for model checkpoints Config
        parser.add_argument('--model_path', default='./checkpoints/sues-200', type=str)

        # Eval before training Config
        parser.add_argument('--zero_shot', default=False, type=bool)

        # Checkpoint to start from Config
        parser.add_argument('--checkpoint_start', default=None)

        # Set num_workers to 0 if on Windows Config
        parser.add_argument('--num_workers', default=0 if os.name == 'nt' else 4, type=int)

        # Train on GPU if available Config
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)

        # For better performance Config
        parser.add_argument('--cudnn_benchmark', default=True, type=bool)

        # Make cudnn deterministic Config
        parser.add_argument('--cudnn_deterministic', default=False, type=bool)

        args = parser.parse_args(namespace=self)


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#
config = Configuration()

if config.dataset == 'U1652-D2S':
    config.query_folder_train = f'./data/{config.dataset_name}/Training/{config.altitude}/satellite'
    config.gallery_folder_train = f'./data/{config.dataset_name}/Training/{config.altitude}/drone'
    config.query_folder_test = f'./data/{config.dataset_name}/Testing/{config.altitude}/query_drone'
    config.gallery_folder_test = f'./data/{config.dataset_name}/Testing/{config.altitude}/gallery_satellite'
elif config.dataset == 'U1652-S2D':
    config.query_folder_train = f'./data/{config.dataset_name}/Training/{config.altitude}/satellite'
    config.gallery_folder_train = f'./data/{config.dataset_name}/Training/{config.altitude}/drone'
    config.query_folder_test = f'./data/{config.dataset_name}/Testing/{config.altitude}/query_satellite'
    config.gallery_folder_test = f'./data/{config.dataset_name}/Testing/{config.altitude}/gallery_drone'

if __name__ == '__main__':

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    
    from sample4geo.hand_convnext.model import make_model

    model = make_model(config)
    print("\nModel:{}".format("adjust model: handcraft convnext-base"))

    # -- print weight config infos   weight_channels
    print(f"\nweight_infonce:{config.weight_infonce}\nweight_fine:{config.weight_fine}\n"
          f"infoNCE_logit:{config.infoNCE_logit}\npos_scale:{config.pos_scale}\n"
          f"weight_channels:{config.weight_channels}\n")
    # print(model)

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


    # Reference Satellite Images
    query_dataset_test = U1652DatasetEval(data_folder=config.query_folder_test,
                                          mode="query",
                                          transforms=val_transforms,
                                          )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size_eval,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    # Query Ground Images Test
    gallery_dataset_test = U1652DatasetEval(data_folder=config.gallery_folder_test,
                                            mode="gallery",
                                            transforms=val_transforms,
                                            sample_ids=query_dataset_test.get_sample_ids(),
                                            gallery_n=config.eval_gallery_n,
                                            )

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)
    
    print(config.gallery_folder_test)
    print(config.query_folder_test)


    print("Query Images Test:", len(query_dataset_test))
    print("Gallery Images Test:", len(gallery_dataset_test))

    # -----------------------------------------------------------------------------#
    # Test Only                                                                    #
    # -----------------------------------------------------------------------------#
import os
import shutil
import numpy as np

if config.only_test:

    print("-------------------My Terst ----------------")
    checkpoint = torch.load(config.ckpt_path)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(config.device)
    model.eval()

    with torch.no_grad():
        test_img = torch.ones((2, 3, 384, 384), dtype=torch.float32)
        test_img = test_img.to(config.device)
        result = model(test_img)
        print(result[0])

    

    print("\n{}[{}]{}".format(30 * "-", "Evaluate", 30 * "-"))
    best_score = 0

    checkpoint = torch.load(config.ckpt_path)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(config.device)

    print("Extract Features:")
    img_features_query, ids_query, path_query = predict(config, model, query_dataloader_test)
    img_features_gallery, ids_gallery, path_gallery = predict(config, model, gallery_dataloader_test)

    # Creo cartelle principali
    os.makedirs("test/queries", exist_ok=True)
    os.makedirs("test/top5_results", exist_ok=True)

    for i, qf in enumerate(img_features_query):
        #print("testing " + path_query[i])
        query_img_path = path_query[i]
        shutil.copy(query_img_path, os.path.join("test/queries", os.path.basename(query_img_path)))

        score = img_features_gallery @ qf.unsqueeze(-1)
        score = score.squeeze().cpu().numpy()
        index = np.argsort(score)[::-1][0:5]
        print(np.sort(score)[::-1][0:10])

        query_name = os.path.splitext(os.path.basename(query_img_path))[0]
        top5_folder = os.path.join("test/top5_results", query_name)
        os.makedirs(top5_folder, exist_ok=True)

        for rank, idx in enumerate(index):
            gallery_img_path = path_gallery[idx]
            shutil.copy(gallery_img_path, os.path.join(top5_folder, f"{rank}.png"))

        
# checkpoint = torch.load(config.ckpt_path, map_location="cpu")
# model.load_state_dict(checkpoint, strict=False)
# model = model.to("cpu")
# model.eval()
# # Supponiamo che le immagini siano [batch, 3, H, W]
# # Metti le dimensioni reali che usa il tuo dataloader
# dummy_input = torch.randn(20, 3, 384, 384)

# onnx_path = "model.onnx"

# torch.onnx.export(
#     model,                       # modello PyTorch
#     dummy_input,                 # input fittizio
#     onnx_path,                   # file di output
#     export_params=True,          # salva i pesi nel file
#     opset_version=17,            # versione ONNX (>=11 di solito va bene)
#     do_constant_folding=True,    # ottimizza costanti
#     input_names=["input"],       # nome degli input
#     output_names=["output"],     # nome degli output
#     dynamic_axes={               # permette batch dinamici
#         "input": {0: "batch_size"},
#         "output": {0: "batch_size"},
#     }
# )

# print(f"Modello esportato in {onnx_path}")
