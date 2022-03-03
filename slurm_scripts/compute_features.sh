#!/bin/bash
#SBATCH --job-name=vic_features
#SBATCH --output=outputs/features/%x.o%j 
#SBATCH --time=8:00:00 
#SBATCH --partition=cpu_long
#SBATCH --mem=32G

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python scripts/compute_features.py $@
