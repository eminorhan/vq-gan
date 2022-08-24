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

python -u /misc/vlgscratch4/LakeGroup/emin/taming-transformers/main.py --base configs/custom_vqgan_32x32_say.yaml -t True --gpus 0,1,2,3 --resume '/misc/vlgscratch4/LakeGroup/emin/taming-transformers/logs/2022-08-07T20-31-31_custom_vqgan_32x32_say'

# python -u /misc/vlgscratch4/LakeGroup/emin/taming-transformers/main.py --base configs/custom_vqgan_32x32_s.yaml -t True --gpus 0,1,2,3 --resume '/misc/vlgscratch4/LakeGroup/emin/taming-transformers/logs/2022-08-15T21-05-50_custom_vqgan_32x32_s'

# python -u /misc/vlgscratch4/LakeGroup/emin/taming-transformers/main.py --base configs/custom_vqgan_32x32_a.yaml -t True --gpus 0,1,2,3 --resume '/misc/vlgscratch4/LakeGroup/emin/taming-transformers/logs/2022-08-17T19-36-46_custom_vqgan_32x32_a'

# python -u /misc/vlgscratch4/LakeGroup/emin/taming-transformers/main.py --base configs/custom_vqgan_32x32_y.yaml -t True --gpus 0,1,2,3 --resume '/misc/vlgscratch4/LakeGroup/emin/taming-transformers/logs/2022-08-18T21-35-31_custom_vqgan_32x32_y'

echo "Done"
