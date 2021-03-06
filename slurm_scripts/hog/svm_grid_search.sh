#!/bin/bash
#SBATCH --job-name=grid_search
#SBATCH --output=outputs/hog/grid_search/%x.o%j 
#SBATCH --time=2:00:00 
#SBATCH --partition=mem
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=50

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -um scripts.hog.svm_grid_search $@
