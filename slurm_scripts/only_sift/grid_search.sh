#!/bin/bash
#SBATCH --job-name=vic_grid
#SBATCH --output=outputs/%x.o%j 
#SBATCH --time=48:00:00 
#SBATCH --partition=cpu_long
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=40

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -u scripts/grid_search.py $@
