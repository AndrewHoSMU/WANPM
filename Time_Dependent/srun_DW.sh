#!/bin/bash
#SBATCH -A cai_deepmartnet_0001 --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrewho@smu.edu
#SBATCH --job-name=WANPM_MP
module load conda
module load gcc/13.2.0
module load cuda/12.4.1-vz7djzz
conda activate torch_gpu
export PYTHONUNBUFFERED=1
python double_well.py
