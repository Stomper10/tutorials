#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TODO: Diffusers scheduler? SD3? (flow matching)
TODO: increase cond_dim if add conditions
"""

import os
import time
import math
import shutil
import logging
import argparse
import datasets
import numpy as np
import pandas as pd
import nibabel as nib
import transformers

from packaging import version
from tqdm.auto import tqdm
from monai import transforms
from torchsummary import summary
from einops_exts import check_shape #, rearrange_many

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torchvision

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers.utils import check_min_version, is_wandb_available #, deprecate, make_image_grid
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr

from diffusers.models.vq_gan_3d import VQGAN #, LPIPS, NLayerDiscriminator, NLayerDiscriminator3D
from diffusers.models.ddpm import Unet3D, GaussianDiffusion
from diffusers.models.ddpm.diffusion import is_list_str #, default
from diffusers.models.ddpm.text import tokenize, bert_embed #, BERT_MODEL_DIM

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")
logger = get_logger(__name__, log_level="INFO")

@torch.no_grad()
def log_validation(input_size, test_dataloader, vae, unet3d, noise_scheduler, accelerator, weight_dtype, gloabal_step, num_samples, save_dir):
    logger.info("Running validation... ")
    vae.eval()
    unet3d.eval()
    vae = accelerator.unwrap_model(vae)
    unet3d = accelerator.unwrap_model(unet3d)

    os.makedirs(os.path.join(save_dir, "generated_volumes"), exist_ok=True)

    images = []
    with torch.autocast(accelerator.device.type, dtype=weight_dtype):
        for i in range(num_samples):
            
            age = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            sex = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            vcf = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            bv = torch.tensor([1.0], dtype=torch.float16).to(unet3d.device)
            cond_tensor = torch.cat([age, sex, vcf, bv], dim=-1)  # Shape: [4]

            image = noise_scheduler.sample(vae=vae, #.cpu().float(),
                                        unet=unet3d,
                                        image_size=int(input_size[1] / vae.config.downsample[1]),
                                        num_frames=int(input_size[0] / vae.config.downsample[0]),
                                        channels=int(vae.config.embedding_dim),
                                        cond=cond_tensor,
                                        batch_size=1
                                        )
            images.append(image)
            image_np = image.squeeze().cpu().numpy()  # Shape: (D, H, W)

            # Save as .npy file
            npy_path = os.path.join(os.path.join(save_dir, "generated_volumes"), f"generated_volume_checkpoint_{gloabal_step}_{i}.npy")
            np.save(npy_path, image_np)
            logger.info(f"Saved generated image to {npy_path}")

        x = next(iter(test_dataloader))["pixel_values"][:num_samples]
    
    images = torch.stack([x.cpu(), torch.cat(images, dim=0).cpu()], dim=1)

    for tracker in accelerator.trackers:
        tracker.name = "wandb"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("Original (left), Reconstruction (right)", np_images, gloabal_step)
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "Original (left), Generation (right) - Sagittal": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, image.shape[2] // 2, :, :]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ],
                    "Original (left), Generation (right) - Coronal": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, :, image.shape[3] // 2, :]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ],
                    "Original (left), Generation (right) - Axial": [
                        wandb.Image(torchvision.utils.make_grid(image[:, :, :, :, image.shape[4] // 2]).permute(1, 2, 0))
                        for _, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del vae
    del unet3d
    del images
    torch.cuda.empty_cache()

def count_activations(model, dummy_input, time, cond):
    activations = []
    def hook_fn(module, input, output):
        # Add the number of elements in the output to the activations list
        activations.append(output.numel())

    # Register a forward hook for each module to capture the output (activations)
    hooks = []
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, 
                              nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.Embedding,)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass through the model
    with torch.no_grad():
        model(dummy_input, time=time, cond=cond)
    
    # Remove all the hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate the total number of activations
    total_activations = sum(activations)
    
    return total_activations

def count_parameters(model):
    """
    Ultimate accurate count of the total, trainable, and non-trainable parameters in a model,
    ensuring shared parameters are not double-counted and handling any complex nested submodules.
    
    Args:
        model (torch.nn.Module): The model to count parameters for.
    
    Returns:
        dict: Contains 'total', 'trainable', 'non_trainable' counts of parameters.
    """
    param_count = {
        "total": 0,
        "trainable": 0,
        "non_trainable": 0,
        "by_layer_type": {},  # Added for detailed reporting by layer type
    }

    # Use a set to track shared parameters and avoid double-counting
    seen_params = set()

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id not in seen_params:
            # Add unique parameter ID to avoid double-counting shared parameters
            seen_params.add(param_id)

            # Count number of elements in the parameter
            num_params = param.numel()

            # Update the total and type-specific counts
            param_count['total'] += num_params
            if param.requires_grad:
                param_count['trainable'] += num_params
            else:
                param_count['non_trainable'] += num_params

            # Track parameters by layer type for detailed reporting
            layer_type = type(param).__name__
            if layer_type not in param_count['by_layer_type']:
                param_count['by_layer_type'][layer_type] = 0
            param_count['by_layer_type'][layer_type] += num_params

            # Optional: print layer-specific details
            print(f"Layer {name}: {num_params} parameters (Trainable: {param.requires_grad})")

    return param_count



def parse_args():
    parser = argparse.ArgumentParser(description="UNET3D training script.")
    parser.add_argument(
        "--axis",
        type=str,
        default=None,
        help=(
            "3D volume's axis that will be processed as the temporal axis."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help=(
            "dim for Unet3D."
        ),
    )
    parser.add_argument(
        "--dim_mults",
        type=str,
        default=None,
        help=(
            "dim_mults for Unet3D."
        ),
    )
    parser.add_argument(
        "--attn_heads",
        type=int,
        default=None,
        help=(
            "attn_heads for Unet3D."
        ),
    )
    parser.add_argument(
        "--attn_dim_head",
        type=int,
        default=None,
        help=(
            "attn_dim_head for Unet3D."
        ),
    )
    parser.add_argument(
        "--pretrained_vae_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--loss_type", type=str, default="l1", help="Loss type: l1 or l2."
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=1000, help="The number of diffusion steps."
    )
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset. Recommended 0.1.")
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_label_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training label. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--valid_label_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the validation label. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=21, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=str,
        default="160,224,160",#512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=16, help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4, # 1.5e-7, # Reference : Waifu-diffusion-v1-4 config # default=4.5e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "piecewise_constant"]'
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.5, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.9, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="vae-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--num_samples", type=int, default=5, help="The number of samples for validation.")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args



# Define custom dataset class
class UKB_Dataset(Dataset):
    """
    A PyTorch Dataset for loading UKB images and labels.
    
    Args:
        image_dir (str): Directory containing image files.
        label_dir (str): CSV file containing image IDs and labels.
        transform (callable, optional): Optional transform to apply to samples.
    """
    def __init__(self, image_dir, label_dir, transform=None, axis="c"):
        super().__init__()
        self.data_dir = image_dir
        data_csv = pd.read_csv(label_dir)
        self.image_names = list(data_csv['rel_path'])
        self.transform = transform
        self.axis = axis
        self.image_paths = [os.path.join(self.data_dir, name) for name in self.image_names]

        # load conditioning variable: age
        self.ages = data_csv['age'].values.astype(np.float16)
        self.min_age, self.max_age = self.ages.min(), self.ages.max()
        self.norm_age = (self.ages - self.min_age) / (self.max_age - self.min_age)
        
        # load conditioning variable: sex
        self.gender = data_csv['sex'].values.astype(np.float16)

        # load conditioning variable: ventricular cerebrospinal fluid
        self.vcf = data_csv['p25004_i2'].values.astype(np.float32)
        self.min_vcf, self.max_vcf = self.vcf.min(), self.vcf.max()
        self.norm_vcf = (self.vcf - self.min_vcf) / (self.max_vcf - self.min_vcf)

        # load conditioning variable: brain volume normalized for head size
        self.bv = data_csv['p25009_i2'].values.astype(np.float32)
        self.min_bv, self.max_bv = self.bv.min(), self.bv.max()
        self.norm_bv = (self.bv - self.min_bv) / (self.max_bv - self.min_bv)

        # load conditioning variable: gender
        #self.genders = data_csv['sex'].values.astype(np.float16)
        # self.gender_encoded = []
        # for gender_int in self.genders:
        #     if gender_int == 1.0: # male
        #         self.gender_encoded.append([1.0])
        #     elif gender_int == 0.0: # female
        #         self.gender_encoded.append([0.0])
        #     else:
        #         self.gender_encoded.append([0.0, 0.0])  # Handle unknown gender
        # self.gender_encoded = np.array(self.gender_encoded, dtype=np.float16)
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
        try:
            image = nib.load(image_path) # (182, 218, 182)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = image.get_fdata()
        image = torch.from_numpy(image) # Stays on CPU # (182, 218, 182)

        axes_mapping = {
            's': (0, 1, 2),
            'c': (1, 0, 2),
            'a': (2, 1, 0)
        }

        try:
            image = image.permute(*axes_mapping[self.axis]).unsqueeze(0) # (1, 218, 182, 182)
        except KeyError:
            raise ValueError("axis must be one of 'a', 'c', or 's'.")

        # Define target size
        # target_size = (128, 128, 128)  # (D₂, H₂, W₂)
        # Resize the volume using trilinear interpolation
        # image = F.interpolate(
        #     image,
        #     size=target_size,
        #     mode='trilinear',
        #     align_corners=False
        # )  # Shape: (1, 1, D₂, H₂, W₂)
        # Remove the batch dimension
        # image = image.squeeze(0).to(torch.float16)  # Shape: (1, D₂, H₂, W₂) # (1, 128, 128, 128)

        age = torch.tensor([self.norm_age[index]], dtype=torch.float16) # Shape: [1]
        sex = torch.tensor([self.gender[index]], dtype=torch.float16)  # Shape: [1]
        vcf = torch.tensor([self.norm_vcf[index]], dtype=torch.float16)  # Shape: [1]
        bv = torch.tensor([self.norm_bv[index]], dtype=torch.float16)  # Shape: [1]
        cond_tensor = torch.cat([age, sex, vcf, bv], dim=-1)  # Shape: [4]

        sample = {
            "pixel_values": image.to(torch.float16),
            "condition": cond_tensor
        }

        if self.transform:
            sample = self.transform(sample)
        del image

        return sample



def main():
    args = parse_args()
    input_size = tuple(int(x) for x in args.resolution.split(","))

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.report_to == "wandb" and accelerator.is_main_process: ###
        wandb.init(project=args.tracker_project_name, resume=True) ###

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load VAE & Unet3D
    vae = VQGAN.from_pretrained( ### set to pretrained VQGAN model path
        args.pretrained_vae_path, subfolder="vqgan", 
    ).to(accelerator.device)
    vae.requires_grad_(False)

    dim_mults = tuple(int(x) for x in args.dim_mults.split(","))
    unet3d = Unet3D(
        dim=args.dim,
        cond_dim=4, # cond_tensor
        dim_mults=dim_mults,
        channels=int(vae.config.embedding_dim),
        attn_heads=args.attn_heads,
        attn_dim_head=args.attn_dim_head,
        resnet_groups=8,
    ).to(accelerator.device)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet3d = Unet3D(
            dim=args.dim,
            cond_dim=4, # cond_tensor
            dim_mults=dim_mults,
            channels=int(vae.config.embedding_dim),
            attn_heads=args.attn_heads,
            attn_dim_head=args.attn_dim_head,
            resnet_groups=8,
        )
        ema_unet3d = EMAModel(
            ema_unet3d.parameters(), 
            decay=0.995,
            update_after_step=2000,
            model_cls=Unet3D, 
            model_config=ema_unet3d.config,
            foreach=args.foreach_ema,
        )

    noise_scheduler = GaussianDiffusion(timesteps=1000,).to(accelerator.device) ###args

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet3d.save_pretrained(os.path.join(output_dir, "unet3d_ema"))

                #logger.info(f"{unet3d = }") # print model architecture
                for _, model in enumerate(models):
                    if isinstance(model, type(accelerator.unwrap_model(unet3d))):
                        model.save_pretrained(os.path.join(output_dir, "unet3d"))
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
                    
                    # make sure to pop weight so that corresponding model is not saved again
                    if weights: ### https://github.com/huggingface/diffusers/issues/2606#issuecomment-1704077101
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet3d_ema"), Unet3D, foreach=args.foreach_ema
                )
                ema_unet3d.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_unet3d.pin_memory()
                else:
                    ema_unet3d.to(accelerator.device)
                del load_model
                # ema_unet3d.to(accelerator.device)
                # del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if isinstance(model, type(accelerator.unwrap_model(unet3d))):
                    load_model = Unet3D.from_pretrained(input_dir, subfolder="unet3d")
                else:
                    raise ValueError(f"unexpected load model: {model.__class__}")

                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes` or `pip install bitsandbytes-windows` for Windows"
            )
        optimizer_cls = bnb.optim.Adam8bit #AdamW8bit
    else:
        optimizer_cls = torch.optim.Adam #AdamW

    optimizer = optimizer_cls(unet3d.parameters(), lr=args.learning_rate)

    # Load data
    train_transforms = transforms.Compose(
        [
            transforms.Resized(keys=["pixel_values"], spatial_size=input_size, size_mode="all"),
            transforms.ScaleIntensityd(keys=["pixel_values"], minv=-1.0, maxv=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["pixel_values"], threshold=-1, above=True, cval=-1.0),
            transforms.ToTensord(keys=["pixel_values"]),
        ]
    )

    with accelerator.main_process_first():
        if args.data_dir is not None: # args.test_data_dir is not None and args.data_dir is not None:
            train_dataset = UKB_Dataset(args.data_dir, args.train_label_dir, transform=train_transforms, axis=args.axis)
            valid_dataset = UKB_Dataset(args.data_dir, args.valid_label_dir, transform=train_transforms, axis=args.axis)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers, #args.train_batch_size*accelerator.num_processes,
        pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.valid_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        power=3.0
    )

    # Prepare everything with our `accelerator`.
    unet3d, optimizer, lr_scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        unet3d, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32 #16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move vae (and ema model) to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    # if args.use_ema:
    #     ema_unet3d.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema:
        if args.offload_ema:
            ema_unet3d.pin_memory()
        else:
            ema_unet3d.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        
    # ------------------------------ TRAIN ------------------------------ #
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test samples = {len(valid_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            # dirs = os.listdir(args.resume_from_checkpoint)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # Model memory check
    # dummy_input = torch.ones(1, 8, 64, 20, 20).float().to(accelerator.device) # 262,144 for 64^3
    # timesteps = torch.randint(0, args.num_timesteps, (dummy_input.shape[0],)).long().to(accelerator.device)
    # cond = torch.tensor([0.5]).float().to(accelerator.device)
    # unet3d.eval()

    # unet3d_activations = count_activations(unet3d, dummy_input, timesteps, cond)
    # print(f"### UNET3D's number of activations: {unet3d_activations:,}")
    
    # Show model architecture and # of params
    param_counts = count_parameters(unet3d)
    print(f"### UNET3D's Total parameters: {param_counts['total']:,}")
    print(f"### UNET3D's Trainable parameters: {param_counts['trainable']:,}")
    print(f"### UNET3D's Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"### UNET3D's Parameters by layer type: {param_counts['by_layer_type']}")
    # summary(unet3d, (8, 32, 32, 32))
    # summary(unet3d, (8, 32, 32, 32, 32)) # assertion
    
    # Training
    if accelerator.is_main_process:
        print("### Start training ###")
    for epoch in range(first_epoch, args.num_train_epochs):
        
        logger.info(f"{epoch = }")
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            unet3d.train()

            # Accumulate gradients
            with accelerator.accumulate(unet3d):
                with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                    x = batch["pixel_values"].to(weight_dtype)

                    with torch.no_grad():
                        z = vae.encode(x, quantize=False, include_embeddings=True)
                        # normalize to -1 and 1 (apply scaling factor)
                        latents = ((z - vae.codebook.embeddings.min()) /
                                   (vae.codebook.embeddings.max() -
                                    vae.codebook.embeddings.min())) * 2.0 - 1.0
                    
                    #print("### latents.shape:", latents.shape, flush=True) # torch.Size([10, 8, 32, 32, 32])
                    B = latents.shape[0]
                    check_shape(latents, 'b c f h w', 
                                c=int(vae.config.embedding_dim), 
                                f=int(input_size[0] / vae.config.downsample[0]), # saggital
                                h=int(input_size[1] / vae.config.downsample[1]), # coronal
                                w=int(input_size[2] / vae.config.downsample[2])) # axial
                    
                    timesteps = torch.randint(0, args.num_timesteps, (B,), device=latents.device).long()

                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device
                        )
                    if args.input_perturbation:
                        new_noise = args.input_perturbation * torch.randn_like(noise)

                    if args.input_perturbation:
                        noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=new_noise)
                    else:
                        noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=noise)
                    #noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=noise)

                    cond = batch["condition"].to(latents.device)
                    if is_list_str(cond):
                        cond = bert_embed(tokenize(cond), return_cls_repr=False)
                        cond = cond.to(latents.device)

                    z_pred = unet3d(x=noisy_latents, time=timesteps, cond=cond, null_cond_prob=0.1)

                    if args.snr_gamma is None:
                        #loss = F.mse_loss(z_pred.float(), noise.float(), reduction="mean")
                        if args.loss_type == 'l1':
                            loss = F.l1_loss(z_pred, noise)
                        elif args.loss_type == 'l2':
                            loss = F.mse_loss(z_pred.float(), noise.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                            dim=1
                        )[0]
                        mse_loss_weights = mse_loss_weights / snr

                        #loss = F.mse_loss(z_pred.float(), noise.float(), reduction="none")
                        if args.loss_type == 'l1':
                            loss = F.l1_loss(z_pred, noise)
                        elif args.loss_type == 'l2':
                            loss = F.mse_loss(z_pred.float(), noise.float(), reduction="mean")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    if not torch.isfinite(loss):
                        logger.info("\nWARNING: non-finite loss.")
                        continue
                        
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet3d.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()

                    gathered_train_loss = accelerator.gather(loss)
                    step_train_loss = gathered_train_loss.sum()

                # if step % 50 == 0:
                logs = {
                    "unet_step_train_loss": step_train_loss.detach().item() / B,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                accelerator.log(logs)

            end_time = time.time()
            if accelerator.is_main_process:
                print(f"### Training step elasped: {end_time - start_time}", flush=True)

            if accelerator.sync_gradients:
                # if args.use_ema:
                #     ema_unet3d.step(unet3d.parameters())
                if args.use_ema:
                    if args.offload_ema:
                        ema_unet3d.to(device="cuda", non_blocking=True)
                    ema_unet3d.step(unet3d.parameters()) # updates EMAModel
                    if args.offload_ema:
                        ema_unet3d.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    try:
                                        shutil.rmtree(removing_checkpoint)
                                    except OSError as e:
                                        logger.error(f"Error removing checkpoint {removing_checkpoint}: {e}")

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    if accelerator.is_main_process:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet3d.store(unet3d.parameters()) # store unet3d weights for later restoration
                            ema_unet3d.copy_to(unet3d.parameters()) # copy ema weights to unet3d
                        # validation
                        print("### Start validation ###")
                        log_validation(input_size, test_dataloader, vae, unet3d, noise_scheduler, accelerator, weight_dtype, global_step, args.num_samples, save_path)
                        # # Switch back to the original UNet parameters.
                        # ema_unet3d.restore(unet3d.parameters()) # restore unet3d weights (from temp saved)
                        
                    with torch.no_grad():
                        vae.eval()
                        unet3d.eval()
                        valid_loss = 0.0
                        for step, batch in enumerate(test_dataloader):
                            with torch.autocast(accelerator.device.type, dtype=weight_dtype):
                                x = batch["pixel_values"].to(weight_dtype)

                                z = vae.encode(x, quantize=False, include_embeddings=True)
                                # normalize to -1 and 1 (apply scaling factor)
                                latents = ((z - vae.codebook.embeddings.min()) /
                                           (vae.codebook.embeddings.max() -
                                            vae.codebook.embeddings.min())) * 2.0 - 1.0
                                
                                B = latents.shape[0]
                                check_shape(latents, 'b c f h w', 
                                            c=int(vae.config.embedding_dim), 
                                            f=int(input_size[0] / vae.config.downsample[0]), # saggital
                                            h=int(input_size[1] / vae.config.downsample[1]), # coronal
                                            w=int(input_size[2] / vae.config.downsample[2])) # axial
                                
                                timesteps = torch.randint(0, args.num_timesteps, (B,), device=latents.device).long()

                                noise = torch.randn_like(latents)
                                if args.noise_offset:
                                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                                    noise += args.noise_offset * torch.randn(
                                        (latents.shape[0], latents.shape[1], 1, 1, 1), device=latents.device
                                    )
                                if args.input_perturbation:
                                    new_noise = args.input_perturbation * torch.randn_like(noise)

                                if args.input_perturbation:
                                    noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=new_noise)
                                else:
                                    noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=noise)
                                #noisy_latents = noise_scheduler.q_sample(x_start=latents, t=timesteps, noise=noise)

                                cond = batch["condition"].to(latents.device)
                                if is_list_str(cond):
                                    cond = bert_embed(tokenize(cond), return_cls_repr=False)
                                    cond = cond.to(latents.device)

                                z_pred = unet3d(x=noisy_latents, time=timesteps, cond=cond, null_cond_prob=0.1)

                                if args.snr_gamma is None:
                                    #loss = F.mse_loss(z_pred.float(), noise.float(), reduction="mean")
                                    if args.loss_type == 'l1':
                                        val_loss = F.l1_loss(z_pred, noise)
                                    elif args.loss_type == 'l2':
                                        val_loss = F.mse_loss(z_pred.float(), noise.float(), reduction="mean")
                                else:
                                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                                    # This is discussed in Section 4.2 of the same paper.
                                    snr = compute_snr(noise_scheduler, timesteps)
                                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                                        dim=1
                                    )[0]
                                    mse_loss_weights = mse_loss_weights / snr

                                    #loss = F.mse_loss(z_pred.float(), noise.float(), reduction="none")
                                    if args.loss_type == 'l1':
                                        val_loss = F.l1_loss(z_pred, noise)
                                    elif args.loss_type == 'l2':
                                        val_loss = F.mse_loss(z_pred.float(), noise.float(), reduction="mean")
                                    val_loss = val_loss.mean(dim=list(range(1, len(val_loss.shape)))) * mse_loss_weights
                                    val_loss = val_loss.mean()

                                if not torch.isfinite(val_loss):
                                    logger.info("\nWARNING: non-finite loss.")
                                    continue

                                # Gather the losses across all processes for logging (if we use distributed training).
                                gathered_valid_loss = accelerator.gather(val_loss)
                                valid_loss += gathered_valid_loss.sum()

                        logs = {
                            "valid_loss": valid_loss.detach().item() / len(test_dataloader.dataset),
                        }
                        accelerator.log(logs)

                    if accelerator.is_main_process:
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet3d.restore(unet3d.parameters()) # restore unet3d weights (from temp saved)

            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()