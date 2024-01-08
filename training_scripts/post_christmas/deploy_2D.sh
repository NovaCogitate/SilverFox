#!/bin/bash
cd /home/j622s/SilverFox
module load cudnn/8.5.0.96/cuda11
module load gmp/6.1.0
module load gcc/11.1.0
path_to_script="/home/j622s/Desktop/SilverFox/training_scripts/post_christmas/2D"

script_name="deploy_256_16.sh"
poetry run python $path_to_script/$script_name

script_name="deploy_256_32.sh"
poetry run python $path_to_script/$script_name

script_name="deploy_512_16.sh"
poetry run python $path_to_script/$script_name