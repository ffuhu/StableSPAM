#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=real_fp4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:01:00
#SBATCH --output=./real_fp4.out

# module purge
# module load 2023
source activate spam
# Your job starts in the directory where you call sbatch

python test_fp4.py --format nvfp4
