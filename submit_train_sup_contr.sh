#!/bin/bash
#SBATCH --job-name=da-office           # Job name
#SBATCH --output=outputs/output_%j.txt      # Output file (%j will be replaced with the job ID)
#SBATCH --error=outputs/error_%j.txt        # Error file (%j will be replaced with the job ID)
#SBATCH --partition=gpu           # Partition to submit the job
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --gres=gpu:1                # Number of GPU cores per task
#SBATCH --mem=10G                    # Memory per node (e.g., 8 GB)
#SBATCH --time=24:00:00             # Time limit (hh:mm:ss)
#SBATCH --mail-type=END,FAIL        # Notifications for job done or fail
#SBATCH --mail-user=gonzalo.quintana@ens-paris-saclay.fr # Email to send notifications to

# Load modules (if applicable)
module load anaconda3/2024.06/gcc-13.2.0

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Avoid OOM error
export TMPDIR=$WORKDIR/tmp

# Activate environment
source .venv/bin/activate

python train_sup_contr.py \
    --num_epochs 700 \
    --image_size 64 \
    --batch_size 512 \
    --num_workers 1 \
    --learning_rate 0.01 \
    --model_name "resnet18" \
    --data_base_dir "data/office" \
    --datalist_pth "data/office/datalist_fold_1.json" \
    --checkpoint_dir "$WORKDIR/ckpt_office/sup_contr_temp_0.05" \
    --default_root_dir $WORKDIR \
    --comet_ml_project_name "contra-da-office" \
    --comet_ml_workspace "gonzaq94" \
    --comet_ml_exp_name "sup_contr_img_size_64_bs_512_temp_0.05_use_projector" \
    --contr_weight_scheduler_step 500 \
    --contr_weight_final 0.0 \
    --contr_temp_steps 500 \
    --init_contr_temp 0.05 \
    --final_contr_temp 0.05 \
    --use_projector \
    --freeze_weights_scheduler_values True False \
    --freeze_weights_scheduler_steps 500 550 
    