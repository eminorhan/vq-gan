#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=376GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=vqgan_saycam
#SBATCH --output=vqgan_saycam_%A_%a.out

module purge
module load cuda-11.4

python -u /misc/vlgscratch4/LakeGroup/emin/taming-transformers/main.py --base configs/custom_vqgan_32x32_imgs.yaml -t True --gpus 0,1,2,3 --resume ''

#python -u /misc/vlgscratch4/LakeGroup/emin/taming-transformers/main.py --base configs/custom_vqgan_32x32.yaml -t True --gpus 0,1,2,3 --resume '/misc/vlgscratch4/LakeGroup/emin/taming-transformers/logs/2022-08-07T20-31-31_custom_vqgan_32x32'

echo "Done"
