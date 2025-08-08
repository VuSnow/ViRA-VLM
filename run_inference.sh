#!/bin/bash -l

#SBATCH --job-name=VLM_desc_inference
#SBATCH --comment="Inference VLM to generate detail description"
#SBATCH --partition=defq
#SBATCH --nodelist=dgx01,dgx02
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16g
#SBATCH --output=logs/inference/%x_%j.out          
#SBATCH --error=logs/inference/%x_%j.err 

# In lệnh và môi trường
pwd
# nvidia-smi
source /home/user05/miniconda3/etc/profile.d/conda.sh
conda activate dungvm2
# /home/user05/miniconda3/envs/dungvm/bin/python3 -m pip install --upgrade pip
# pip install -r requirements.txt --user
# python -m stage1_description.dataset.dataset_cleaning

# Dùng tee để vừa ghi log vừa in ra terminal
python -m stage1_description.inference 
