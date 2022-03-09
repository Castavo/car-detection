#!/bin/bash
#SBATCH --job-name=compute_features
#SBATCH --output=outputs/hog/%x.o%j 
#SBATCH --time=1:00:00 
#SBATCH --partition=cpu_short
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=40

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -um scripts.hog.compute_features
