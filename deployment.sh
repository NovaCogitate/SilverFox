
# Check if the home is /home/pedro
if [[ $HOME == "/home/pedro" ]]; then
    data_folder="/home/pedro/Desktop/Repos/SilverFox_data/numpy_dataset_2D_128_classes"
    data_folder_3D="/home/pedro/Desktop/Repos/SilverFox_data/3D_data"
    result_output="/home/pedro/Desktop/Repos/SilverFox_data/output"
elif [[ $HOME == "/home/j622s" ]]; then
    data_folder="/dkfz/cluster/gpu/data/OE0094/j622s/all_over_again_post_crisis/3D_data"
    data_folder_3D="/dkfz/cluster/gpu/data/OE0094/j622s/all_over_again_post_crisis/numpy_dataset_2D_128_classes"
    result_output="/dkfz/cluster/gpu/data/OE0094/j622s/SilverOut_results"
else
    echo "Unknown home directory: $HOME"
    exit 1
fi



# Check if the folder exists
if [ -d "$result_output/results_2D" ]; then
    echo "Folder exists."
else
    echo "Folder does not exist."
fi


# Set Docker and cluster options
export LSB_CONTAINER_IMAGE=shawnpedro/silverfox:latest
export CLUSTER_CONTAINER_OPTIONS=" --volume=$result_output/results_2D:/app/results_2D --volume=$result_output/results:/app/results --volume=$result_output/results_3D:/app/results_3D --volume=$data_folder:/app/dataset/numpy_dataset_2D_128_classes --volume=$data_folder_3D:/app/dataset/3D_data" 


$result_output/results_2D
$result_output/results
$result_output/results_3D
$data_folder
$data_folder_3D

export LSB_CONTAINER_IMAGE=shawnpedro/silverfox:latest
export CLUSTER_CONTAINER_OPTIONS=" --bind=$result_output/results_2D:/results_2D --bind=$result_output/results:/results --bind=$result_output/results_3D:/results_3D --bind=$data_folder:/numpy_dataset_2D_128_classes --bind=$data_folder_3D:/3D_data" 


# Set memory requirement
memory="20G"

# Submit job to the cluster
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -q gpu -R "rusage[mem=20GB]" --env "CLUSTER_CONTAINER_OPTIONS=$CLUSTER_CONTAINER_OPTIONS" poetry run python /app/docker_paths_mount_and_output_test.py




I want to bind the following folders to the corresponding location when submitting a bsub using appimage

$result_output/results_2D -> /app/results_2D
$result_output/results -> /app/results
$result_output/results_3D -> /app/results_3D
$data_folder -> /app/dataset/numpy_dataset_2D_128_classes
$data_folder_3D -> /app/dataset/3D_data



# Define the container image
export LSB_CONTAINER_IMAGE=shawnpedro/silverfox:latest


# Define container bind options
export CLUSTER_CONTAINER_OPTIONS=" --bind=$data_folder:/numpy_dataset_2D_128_classes \
                                   --bind=$data_folder_3D:/3D_data"



export LSB_CONTAINER_IMAGE="/dkfz/cluster/gpu/data/OE0094/j622s/apptaimer/pulldir/silverfox_latest.sif"
# Define container bind options
export CLUSTER_CONTAINER_OPTIONS=" --bind=$result_output/results_2D:/results_2D \
                                    --bind=$result_output/results:/results \
                                    --bind=$result_output/results_3D:/results_3D \
                                    --bind=$data_folder:/numpy_dataset_2D_128_classes \
                                    --bind=$data_folder_3D:/3D_data"

# Set memory requirement
memory="20G"

# Submit job to the cluster
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory \
     -q gpu -R "rusage[mem=20GB]" \
     --env "CLUSTER_CONTAINER_OPTIONS=$CLUSTER_CONTAINER_OPTIONS" \
     poetry run python /app/docker_paths_mount_and_output_test.py



export LSB_CONTAINER_IMAGE="/dkfz/cluster/gpu/data/OE0094/j622s/apptaimer/pulldir/silverfox_latest.sif"



















































 

data_folder="/dkfz/cluster/gpu/data/OE0094/j622s/all_over_again_post_crisis/3D_data"
data_folder_3D="/dkfz/cluster/gpu/data/OE0094/j622s/all_over_again_post_crisis/numpy_dataset_2D_128_classes"
result_output="/dkfz/cluster/gpu/data/OE0094/j622s/SilverOut_results"

export CLUSTER_CONTAINER_OPTIONS="--bind=$result_output/results_2D:/app/results_2D \
                                  --bind=$result_output/results:/app/results \
                                  --bind=$result_output/results_3D:/app/results_3D \
                                  --bind=$data_folder:/app/dataset/numpy_dataset_2D_128_classes \
                                  --bind=$data_folder_3D:/app/dataset/3D_data"

export LSB_CONTAINER_IMAGE="/dkfz/cluster/gpu/data/OE0094/j622s/apptaimer/pulldir/silverfox_latest.sif" 


# Set memory requirement
memory="20G"

# Submit the job
bsub -q gpu -R "rusage[mem=$memory]" -app apptainer-generic \
     --env "CLUSTER_CONTAINER_OPTIONS=$CLUSTER_CONTAINER_OPTIONS" \
     poetry run python /app/docker_paths_mount_and_output_test.py


apptainer shell /dkfz/cluster/gpu/data/OE0094/j622s/apptaimer/pulldir/silverfox_latest.sif

# Set memory requirement
memory="20G"

# set the image path 
export LSB_CONTAINER_IMAGE="/dkfz/cluster/gpu/data/OE0094/j622s/apptaimer/container_2.sif"

# set the container options
result_output="/dkfz/cluster/gpu/data/OE0094/j622s/SilverOut_results"
command="/app/.venv/bin/python -c \"import torch; print('Torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())\""
export CLUSTER_CONTAINER_OPTIONS="--nv"

# Submit the job if the variable $CLUSTER_CONTAINER_OPTIONS is set 
if [ -z ${CLUSTER_CONTAINER_OPTIONS+x} ]; then 
    bsub -q gpu -R "rusage[mem=$memory]" -app apptainer-generic "$command"
else 
    echo "CLUSTER_CONTAINER_OPTIONS is set to '$CLUSTER_CONTAINER_OPTIONS'"
    bsub -q gpu -R "rusage[mem=$memory]" -app apptainer-generic \
         --env "CLUSTER_CONTAINER_OPTIONS=$CLUSTER_CONTAINER_OPTIONS" \
        "$command"
fi





# Set memory requirement
memory="20G"

# Set the image path 
export LSB_CONTAINER_IMAGE="/dkfz/cluster/gpu/data/OE0094/j622s/apptaimer/container_2.sif"

# Set the container options
command="/app/.venv/bin/python -c \"import torch; print('Torch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available())\""

# Submit the job
bsub -q gpu -R "rusage[mem=$memory]" -app apptainer-generic "$command"
