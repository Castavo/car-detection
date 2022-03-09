#!/bin/bash
#SBATCH --job-name=vic_train
#SBATCH --output=outputs/hog/train/%x.o%j 
#SBATCH --time=4:00:00 
#SBATCH --partition=cpu_med
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=40

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -um scripts.hog.train_model $@
