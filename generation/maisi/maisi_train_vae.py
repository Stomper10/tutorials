import argparse
#import glob
import time
import json
import math
import os
#import tempfile
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

import torch
from monai.networks.nets import PatchDiscriminator
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
#import wandb

from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image

import warnings

warnings.filterwarnings("ignore")

print_config()

def main():
    parser = argparse.ArgumentParser(description="MAISI VAE training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the data."
        ),
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default=None,
        help=(
            "A file containing the train config."
        ),
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help=(
            "A file containing the model config."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    args = parser.parse_args()
    #args = argparse.Namespace()
    # args.output_dir = "/data/wonyoungjang/tutorials/generation/maisi/results/E0_VAE_maisi" # SLURM_JOBNAME
    # args.data_dir = "/data/wonyoungjang/20252_unzip"
    # args.train_config_path = "/data/wonyoungjang/tutorials/generation/maisi/configs/config_maisi_vae_train.json"
    # args.model_config_path = "/data/wonyoungjang/tutorials/generation/maisi/configs/config_maisi.json"
    #args.model_dir = os.path.join(args.output_dir, "models")
    args.tfevent_dir = os.path.join(args.output_dir, "tfevent")
    # args.max_train_steps = 1000000
    args.gradient_accumulation_steps = 1


    # Data
    train_label_df = pd.read_csv("/data/wonyoungjang/tutorials/generation/maisi/data/train.csv")
    valid_label_df = pd.read_csv("/data/wonyoungjang/tutorials/generation/maisi/data/valid.csv")

    train_files = [{"image": os.path.join(args.data_dir, image_name)} for image_name in train_label_df["rel_path"]]
    valid_files = [{"image": os.path.join(args.data_dir, image_name)} for image_name in valid_label_df["rel_path"]]

    # Expandable to more datasets
    datasets = {
        1: {
            "data_name": "T1_brain_to_MNI",
            "train_files": train_files,
            "val_files": valid_files,
            "modality": "mri",
        }
    }



    # Read in env setting
    # model path
    # Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    # trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    # trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    # print(f"Trained model will be saved as {trained_g_path} and {trained_d_path}.")

    # initialize tensorboard writer
    #wandb.init(project=args.tracker_project_name, resume=True)
    Path(args.tfevent_dir).mkdir(parents=True, exist_ok=True)
    tensorboard_path = os.path.join(args.tfevent_dir, "autoencoder")
    Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_path)
    print(f"Tensorboard event will be saved as {tensorboard_path}.")



    # Read config settings
    config_dict = json.load(open(args.model_config_path, "r"))
    for k, v in config_dict.items():
        setattr(args, k, v)

    # check the format of inference inputs
    config_train_dict = json.load(open(args.train_config_path, "r"))
    for k, v in config_train_dict["data_option"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")
    for k, v in config_train_dict["autoencoder_train"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    print("Network definition and training hyperparameters have been loaded.")



    # set seed
    set_determinism(seed=0)



    # Build training dataset
    # Initialize file lists
    train_files = {"mri": []}
    val_files = {"mri": []}

    # Function to add assigned class to datalist
    def add_assigned_class_to_datalist(datalist, classname):
        for item in datalist:
            item["class"] = classname
        return datalist

    # Process datasets
    for _, dataset in datasets.items():
        train_files_i = dataset["train_files"]
        val_files_i = dataset["val_files"]
        print(f"{dataset['data_name']}: number of training data is {len(train_files_i)}.")
        print(f"{dataset['data_name']}: number of val data is {len(val_files_i)}.")

        # attach modality to each file
        modality = dataset["modality"]
        train_files[modality] += add_assigned_class_to_datalist(train_files_i, modality)
        val_files[modality] += add_assigned_class_to_datalist(val_files_i, modality)

    # Print total numbers for each modality
    for modality in train_files.keys():
        print(f"Total number of training data for {modality} is {len(train_files[modality])}.")
        print(f"Total number of val data for {modality} is {len(val_files[modality])}.")

    # Combine the data
    train_files_combined = train_files["mri"]
    val_files_combined = val_files["mri"]



    # Define data transforms
    train_transform = VAE_Transform(
        is_train=True,
        random_aug=args.random_aug,  # whether apply random data augmentation for training
        k=4,  # patches should be divisible by k
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float16,  # final data type
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,  # patches should be divisible by k
        val_patch_size=args.val_patch_size,  # if None, will validate on whole image volume
        output_dtype=torch.float16,  # final data type
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )



    # Build dataloader
    print(f"Total number of training data is {len(train_files_combined)}.")
    dataset_train = CacheDataset(data=train_files_combined, transform=train_transform, cache_rate=args.cache, num_workers=8)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    print(f"Total number of validation data is {len(val_files_combined)}.")
    dataset_val = CacheDataset(data=val_files_combined, transform=val_transform, cache_rate=args.cache, num_workers=8)
    dataloader_val = DataLoader(dataset_val, batch_size=args.val_batch_size, num_workers=4, shuffle=False)



    # Init networks
    device = torch.device("cuda")

    args.autoencoder_def["num_splits"] = 1
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)



    # Training config
    # config loss and loss weight
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
        print("Use l2 loss")
    else:
        intensity_loss = L1Loss(reduction="mean")
        print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    loss_perceptual = (
        PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)
    )

    # config optimizer and lr scheduler
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)


    # please adjust the learning rate warmup rule based on your dataset and n_epochs
    def warmup_rule(epoch):
        # learning rate warmup rule
        if epoch < 10:
            return 0.01
        elif epoch < 20:
            return 0.1
        else:
            return 1.0


    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    # set AMP scaler
    if args.amp:
        # test use mean reduction for everything
        scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
        scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)



    # Load ckpt
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.n_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.n_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.n_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0

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
            print(f"Resuming from checkpoint {path}")
            torch.load(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )
    progress_bar.set_description("Steps")





    # Training
    # Initialize variables
    # val_interval = args.val_interval
    best_val_recon_step_loss = 10000000.0
    # total_step = 0
    # start_epoch = 0
    max_epochs = args.n_epochs

    # Setup validation inferer
    val_inferer = (
        SlidingWindowInferer(
            roi_size=args.val_sliding_window_patch_size,
            sw_batch_size=1,
            progress=False,
            overlap=0.0,
            device=torch.device("cpu"),
            sw_device=device,
        )
        if args.val_sliding_window_patch_size
        else SimpleInferer()
    )


    def loss_weighted_sum(losses):
        return losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + args.perceptual_weight * losses["p_loss"]


    # Training and validation loops
    for epoch in range(first_epoch, max_epochs):
        print("lr:", scheduler_g.get_lr())
        autoencoder.train()
        discriminator.train()
        train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

        for batch in dataloader_train:
            start_time = time.time()
            images = batch["image"].to(device).contiguous()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                # Train Generator
                reconstruction, z_mu, z_sigma = autoencoder(images)
                losses = {
                    "recons_loss": intensity_loss(reconstruction, images),
                    "kl_loss": KL_loss(z_mu, z_sigma),
                    "p_loss": loss_perceptual(reconstruction.float(), images.float()),
                }
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

                if args.amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizer_g)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    optimizer_g.step()

                # Train Discriminator
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                if args.amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    optimizer_d.step()

            # Log training loss
            end_time = time.time()
            progress_bar.update(1)
            global_step += 1
            #total_step += 1
            for loss_name, loss_value in losses.items():
                tensorboard_writer.add_scalar(f"train_{loss_name}_iter", loss_value.item(), global_step)
                train_epoch_losses[loss_name] += loss_value.item()
            tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, global_step)
            tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, global_step)
            tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, global_step)

            if global_step % args.checkpointing_steps == 0:
                trained_g_path = os.path.join(args.output_dir, f"checkpoint-{global_step}/autoencoder.pt")
                trained_d_path = os.path.join(args.output_dir, f"checkpoint-{global_step}/discriminator.pt")
                torch.save(autoencoder.state_dict(), trained_g_path)
                torch.save(discriminator.state_dict(), trained_d_path)
                print("Save trained autoencoder.")
                print("Save trained discriminator.")

                # Validation
                #if epoch % val_interval == 0:
                autoencoder.eval()
                val_step_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
                val_loader_iter = iter(dataloader_val)
                for batch in dataloader_val:
                    with torch.no_grad():
                        with autocast("cuda", enabled=args.amp):
                            images = batch["image"]
                            reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, images)
                            reconstruction = reconstruction.to(device)
                            val_step_losses["recons_loss"] += intensity_loss(reconstruction, images.to(device)).item()
                            val_step_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                            val_step_losses["p_loss"] += loss_perceptual(reconstruction, images.to(device)).item()

                for key in val_step_losses:
                    val_step_losses[key] /= len(dataloader_val)

                val_loss_g = loss_weighted_sum(val_step_losses)
                print(f"Step {global_step} val_vae_loss {val_loss_g}: {val_step_losses}.")

                if val_loss_g < best_val_recon_step_loss:
                    best_val_recon_step_loss = val_loss_g
                    trained_g_path_step = f"{trained_g_path[:-3]}_step{global_step}.pt"
                    torch.save(autoencoder.state_dict(), trained_g_path_step)
                    print("Got best val vae loss.")
                    print("Save trained autoencoder to", trained_g_path_step)

                for loss_name, loss_value in val_step_losses.items():
                    tensorboard_writer.add_scalar(loss_name, loss_value, global_step)

                # Monitor scale_factor
                # We'd like to tune kl_weights in order to make scale_factor close to 1.
                scale_factor_sample = 1.0 / z_mu.flatten().std()
                tensorboard_writer.add_scalar("val_one_sample_scale_factor", scale_factor_sample, global_step)

                # Monitor reconstruction result
                center_loc_axis = find_label_center_loc(images[0, 0, ...])
                vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
                vis_recon_image = get_xyz_plot(reconstruction[0, ...], center_loc_axis, mask_bool=False)

                tensorboard_writer.add_image(
                    "val_orig_img",
                    vis_image.transpose([2, 0, 1]),
                    global_step,
                )
                tensorboard_writer.add_image(
                    "val_recon_img",
                    vis_recon_image.transpose([2, 0, 1]),
                    global_step,
                )

                # show_image(vis_image, title="val image")
                # show_image(vis_recon_image, title="val recon result")

        scheduler_g.step()
        scheduler_d.step()
        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(dataloader_train)
        print(f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses)}: {train_epoch_losses}.")
        for loss_name, loss_value in train_epoch_losses.items():
            tensorboard_writer.add_scalar(f"train_{loss_name}_epoch", loss_value, epoch)

if __name__ == "__main__":
    main()