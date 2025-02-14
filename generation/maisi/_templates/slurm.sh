#!/bin/bash

#SBATCH --job-name=E0_UNET3D_192
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --exclude=b07,b30
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:SIGUSR1@30
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/outputs/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc
source /home/s1/wonyoungjang/anaconda3/bin/activate
conda activate diff

max_restarts=20
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)

function resubmit()
{
    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximum restarts limit"
        exit 1
    fi
}

trap 'resubmit' SIGUSR1

{
echo "Training LDM (2+1)D UNET from scratch."
echo "'c': (3, 1, 0, 2)"

export JOB_NAME=$SLURM_JOB_NAME
export VAE_PATH="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/E0_VQGAN3D_192/checkpoint-xxxxxx"

accelerate launch --config_file /shared/s1/lab06/wonyoung/diffusers/sd3/config/config_single.yaml \
    /shared/s1/lab06/wonyoung/diffusers/sd3/LDM/train_unet3d.py \
    --pretrained_vae_path=$VAE_PATH \
    --data_dir="/leelabsg/data/20252_unzip" \
    --train_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/data/valid.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/diffusers/sd3/LDM/results/$JOB_NAME" \
    --resume_from_checkpoint="latest" \
    --axis="c" \
    --dim=128 \
    --dim_mults="1,2,4,8" \
    --attn_heads=8 \
    --attn_dim_head=64 \
    --seed=42 \
    --allow_tf32 \
    --max_grad_norm=1 \
    --mixed_precision="fp16" \
    --dataloader_num_workers=4 \
    --tracker_project_name=$JOB_NAME \
    --resolution="224,48,48" \
    --learning_rate=1e-5 \
    --scale_lr \
    --lr_scheduler="polynomial" \
    --gradient_accumulation_steps=2 \
    --train_batch_size=2 \
    --valid_batch_size=2 \
    --max_train_steps=1000000 \
    --checkpointing_steps=5000 \
    --num_samples=1 \
    --loss_type="l2" \
    --noise_offset=0.1 \
    --snr_gamma=5.0 \
    --num_timesteps=1000 \
    --report_to="wandb"
} &
wait
exit 0
