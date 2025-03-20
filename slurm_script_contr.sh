#!/bin/bash
#SBATCH --job-name=contr-da-resnet           # Job name
#SBATCH --output=outputs/output_%j.txt      # Output file (%j will be replaced with the job ID)
#SBATCH --error=outputs/error_%j.txt        # Error file (%j will be replaced with the job ID)
#SBATCH --partition=gpu           # Partition to submit the job
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --gres=gpu:1                # Number of GPU cores per task
#SBATCH --mem=8G                    # Memory per node (e.g., 8 GB)
#SBATCH --time=4:00:00             # Time limit (hh:mm:ss)
#SBATCH --mail-type=END,FAIL        # Notifications for job done or fail
#SBATCH --mail-user=gonzalo.quintana@ens-paris-saclay.fr # Email to send notifications to

# Load modules (if applicable)
#module load python/3.9

# Activate environment
source .venv/bin/activate

# Run your command or script
python train_office_resnet_contr.py

