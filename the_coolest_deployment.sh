# Set memory requirement
memory="20G"

# Set the image path 
export LSB_CONTAINER_IMAGE="/dkfz/cluster/gpu/data/OE0094/j622s/apptaimer/container_2.sif"

# Set the container options
result_output="/dkfz/cluster/gpu/data/OE0094/j622s/SilverOut_results"
# export CLUSTER_CONTAINER_OPTIONS="--nv"

module load cudnn/8.5.0.96/cuda11
module load gmp/6.1.0  
module load gcc/11.1.0
# Command to check GPU and CUDA versions
gpu_check_command="nvidia-smi && nvcc -V && /app/.venv/bin/python -c \"import torch; print('Torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())\""

# Submit the job
echo "CLUSTER_CONTAINER_OPTIONS is set to '$CLUSTER_CONTAINER_OPTIONS'"
bsub -q gpu -R "rusage[mem=$memory]" -app apptainer-generic \
#      --env "CLUSTER_CONTAINER_OPTIONS=$CLUSTER_CONTAINER_OPTIONS" \
     "$gpu_check_command"
