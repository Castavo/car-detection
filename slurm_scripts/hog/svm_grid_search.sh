#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=outputs/hog/grid_search/%x.o%j 
#SBATCH --time=12:00:00 
#SBATCH --partition=cpu_long
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=20

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -um scripts.hog.svm_grid_search $@
