#!/bin/bash
#SBATCH --job-name=vic_test_features
#SBATCH --output=outputs/test_features/%x.o%j 
#SBATCH --time=4:00:00 
#SBATCH --partition=cpu_med
#SBATCH --mem=32G

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -u scripts/compute_test_features.py $@
