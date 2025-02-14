#!/bin/sh
#SBATCH --job-name=E0_VAE_MAISI
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu-farm
#SBATCH --mem=125GB
#SBATCH --cpus-per-task=32
#SBATCH --qos=high_gpu_users
#SBATCH -e %x_%j.err#SBATCH
#SBATCH -o /data/wonyoungjang/tutorials/generation/maisi/outputs/%x-%j.txt

source ${HOME}/.bashrc
source /opt/ohpc/pub/anaconda3/etc/profile.d/conda.sh
conda activate diff

export CUDA_HOME=$CONDA_PREFIX
export JOB_NAME=$SLURM_JOB_NAME

echo $(date +"%Y-%m-%d %H-%M-%S")
echo "Training MAISI VAE from scratch."

python3 /data/wonyoungjang/tutorials/generation/maisi/maisi_train_vae.py \
    --output_dir="/data/wonyoungjang/tutorials/generation/maisi/results/$JOB_NAME" \
    --data_dir="/data/wonyoungjang/20252_unzip" \
    --train_config_path="/data/wonyoungjang/tutorials/generation/maisi/configs/config_maisi_vae_train.json" \
    --model_config_path="/data/wonyoungjang/tutorials/generation/maisi/configs/config_maisi.json" \
    --max_train_steps=100 \
    --checkpointing_steps=10 \
    --resume_from_checkpoint="latest"

exit 0