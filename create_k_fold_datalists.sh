#!/bin/bash
#SBATCH --job-name=synthetic-patch-generation           # Job name
#SBATCH --output=outputs/output_%j.txt      # Output file (%j will be replaced with the job ID)
#SBATCH --error=outputs/error_%j.txt        # Error file (%j will be replaced with the job ID)
#SBATCH --partition=cpu_short           # Partition to submit the job
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=2G                    # Memory per node (e.g., 8 GB)
#SBATCH --time=0:10:00             # Time limit (hh:mm:ss)

# Load modules (if applicable)
module load anaconda3/2024.06/gcc-13.2.0

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate environment
source .venv/bin/activate

# Run your command or script
python src/data/create_k_fold_datalists.py \
    --data_dir "data/office" \
    --k_folds 5 \
    --output_dir  "data/office"