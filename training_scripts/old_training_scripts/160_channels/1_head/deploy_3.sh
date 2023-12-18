#!/bin/bash
cd /home/j622s/alien_repo
module load cudnn/8.5.0.96/cuda11
module load gmp/6.1.0
module load gcc/11.1.0
source /dkfz/cluster/gpu/data/OE0094/j622s/poetry_home/venv/bin/activate 
path_to_script="/home/j622s/SilverFox/training_scripts/old_training_scripts/160_channels/1_head/"
script_name="train_alien_test_2D_256_16,8.py"
poetry run python $path_to_script/$script_name