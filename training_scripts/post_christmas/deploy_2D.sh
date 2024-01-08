#!/bin/bash
cd /home/j622s/SilverFox
memory="24G"
path_to_script="/home/j622s/Desktop/SilverFox/training_scripts/post_christmas/2D"

script_name="deploy_256_16.sh"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu bash $path_to_script/$script_name

script_name="deploy_256_32.sh"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu bash $path_to_script/$script_name

script_name="deploy_512_16.sh"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu bash $path_to_script/$script_name