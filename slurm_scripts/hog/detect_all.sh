#!/bin/bash
#SBATCH --job-name=detect
#SBATCH --output=outputs/hog/detect/%x.o%j 
#SBATCH --time=2:00:00 
#SBATCH --partition=mem
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=70

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -um scripts.hog.detect_all $@
