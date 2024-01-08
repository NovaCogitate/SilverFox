#!/bin/bash
cd /home/j622s/SilverFox
module load cudnn/8.5.0.96/cuda11
module load gmp/6.1.0
module load gcc/11.1.0
path_to_script="/home/j622s/SilverFox/training_scripts/post_christmas/2D"
script_name="train_alien_test_2D_256_32,16.py"
/home/j622s/.local/bin/poetry run python $path_to_script/$script_name