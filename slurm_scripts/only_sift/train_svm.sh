#!/bin/bash
#SBATCH --job-name=vic_train
#SBATCH --output=outputs/train/%x.o%j 
#SBATCH --time=12:00:00 
#SBATCH --partition=cpu_long
#SBATCH --mem=128G
#SBATCH --mail-user=baptiste.prevot@student-cs.fr

# Module load 
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment code
source activate opencv

# Train the network
python -u scripts/train_svm.py $@
