#!/bin/bash
#SBATCH --job-name=singularity_gpu_job
#SBATCH --error=error_%j.txt
#SBATCH --output=output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=20-00
#SBATCH --gres=gpu:1

python setup.py install --user