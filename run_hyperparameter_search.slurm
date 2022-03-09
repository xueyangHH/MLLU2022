#!/bin/bash
#SBATCH --job-name=roberta_hyperparameter_search
#SBATCH --open-mode=append
#SBATCH --output=/scratch/xh1128/hw3/hw3.out
#SBATCH --error=/scratch/xh1128/hw3/hw3.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --mail-user=<xh1128@nyu.edu>
#SBATCH --mail-type=ALL

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.
singularity exec --nv --bind /scratch/$USER --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/hw3
python run_hyperparameter_search.py /scratch/xh1128/hw3/BoolQ
"
