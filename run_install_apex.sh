module load cudnn/8.5.0.96/cuda11
module load gmp/6.1.0
module load gcc/11.1.0

silver_path="/home/j622s/SilverFox"
cd $silver_path
memory="24G"
poetry_path="/dkfz/cluster/gpu/data/OE0094/j622s/poetry_home/bin/poetry"

$poetry_path run python ./install_apex.py 