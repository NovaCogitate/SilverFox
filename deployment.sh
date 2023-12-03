
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

# Set Docker and cluster options
export LSB_CONTAINER_IMAGE=shawnpedro/silverfox:latest
export CLUSTER_CONTAINER_OPTIONS=" --volume=$result_output/results_2D:/app/results_2D --volume=$result_output/results:/app/results --volume=$result_output/results_3D:/app/results_3D --volume=$data_folder:/app/dataset/numpy_dataset_2D_128_classes --volume=$data_folder_3D:/app/dataset/3D_data " 

# Set memory requirement
memory="20G"

# Submit job to the cluster
bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=$memory -q gpu -R "rusage[mem=20GB]" --env "CLUSTER_CONTAINER_OPTIONS=$CLUSTER_CONTAINER_OPTIONS" poetry run python /app/docker_paths_mount_and_output_test.py
