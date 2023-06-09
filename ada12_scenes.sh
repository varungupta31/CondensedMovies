#!/bin/bash
#SBATCH -A varungupta
#SBATCH -n 1
#SBATCH -w gnode012
#SBATCH --gres gpu:1
#SBATCH --mem=8G
#SBATCH --time=INFINITE
#SBATCH --mail-type=END
#SBATCH --mail-user=varungupta.iiith@gmail.com

conda init --all
source activate cmd-chall
module load u18/cudnn/8.4.0-cuda-11.6
module load u18/cuda/11.6

python train.py configs/moe.json
