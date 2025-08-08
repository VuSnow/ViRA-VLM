#!/bin/bash -l

#SBATCH --job-name=VLM_desc_training
#SBATCH --comment="Training VLM to generate detail description"
#SBATCH --partition=defq
#SBATCH --nodelist=dgx01,dgx02
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16g
#SBATCH --output=logs/training/%x_%j.out          
#SBATCH --error=logs/training/%x_%j.err 

# In lệnh và môi trường
# pwd
nvidia-smi
source /home/user05/miniconda3/etc/profile.d/conda.sh
conda activate dungvm2
# unset PYTHONPATH
# /home/user05/miniconda3/envs/dungvm/bin/python3 -m pip install --upgrade pip
# pip install -r requirements.txt --user
# python -m stage1_description.dataset.dataset_cleaning

# Dùng tee để vừa ghi log vừa in ra terminal
python -m stage1_description.train \
    --config_path "/home/user05/dungvm/configs/configs.yaml" \
    --dataset_name "5CD-AI/Viet-LAION-Gemini-VQA" \
    --freeze_llm False \
    --freeze_vision False \
    --num_samples 800000 \
    --split_ratio 0.1 \
    --seed 42 2>&1 | tee logs/VLM_desc_$(date +%Y%m%d_%H%M%S).log
