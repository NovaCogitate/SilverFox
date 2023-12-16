# remote server goto folder 
silver_path="/home/j622s/SilverFox"
cd $silver_path
memory="24G"
poetry_path="/dkfz/cluster/gpu/data/OE0094/j622s/poetry_home/bin/poetry"

# 1 head
path_to_script="/home/j622s/SilverFox/training_scripts/old_training_scripts/96_channels/1_head"
script_name="train_alien_test_2D_256_16,8.py"

# script_name="train_alien_test_2D_256_8.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

# script_name="train_alien_test_2D_256_16.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16,8.py"
# source /dkfz/cluster/gpu/data/OE0094/j622s/poetry_home/venv/bin/activate &&  
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $path_to_script/deploy_3.sh

# 2 head 
path_to_script="/home/j622s/SilverFox/training_scripts/old_training_scripts/96_channels/2_head"

# script_name="train_alien_test_2D_256_8.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

# script_name="train_alien_test_2D_256_16.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16,8.py"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

# 3 head 
path_to_script="/home/j622s/SilverFox/training_scripts/old_training_scripts/96_channels/3_head"

# script_name="train_alien_test_2D_256_8.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16.py"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16,8.py"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name


# for more channels (160_channels)

# 1 head
path_to_script="/home/j622s/SilverFox/training_scripts/old_training_scripts/160_channels/1_head"

# script_name="train_alien_test_2D_256_8.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

# script_name="train_alien_test_2D_256_16.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16,8.py"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

# 2 head 
path_to_script="/home/j622s/SilverFox/training_scripts/old_training_scripts/160_channels/2_head"

# script_name="train_alien_test_2D_256_8.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

# script_name="train_alien_test_2D_256_16.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16,8.py"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

# 3 head 
path_to_script="/home/j622s/SilverFox/training_scripts/old_training_scripts/160_channels/3_head"

# script_name="train_alien_test_2D_256_8.py"
# bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16.py"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name

script_name="train_alien_test_2D_256_16,8.py"
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -R "rusage[mem=12GB]" -L /bin/bash -q gpu $poetry_path run python $path_to_script/$script_name
