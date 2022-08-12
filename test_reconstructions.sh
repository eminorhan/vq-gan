#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=vqgan_test_reconstructions
#SBATCH --output=vqgan_test_reconstructions_%A_%a.out

module purge
module load cuda-11.4

python -u /misc/vlgscratch4/LakeGroup/emin/taming-transformers/test_reconstructions.py

echo "Done"
