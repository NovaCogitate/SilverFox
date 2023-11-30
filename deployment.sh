data_folder="/home/pedro/Desktop/Repos/SilverFox_data/numpy_dataset_2D_128_classes"
data_folder_3D="/home/pedro/Desktop/Repos/SilverFox_data/3D_data"

result_output="/home/pedro/Desktop/Repos/SilverFox_data/output"


docker run -it --gpus all -v $($result_output)/results_2D:/app/results_2D -v $($result_output)/results:/app/results -v $($result_output)/results_3D:/app/results_3D docker run -it --gpus all -v $(pwd):/app/dataset/numpy_dataset_2D_128_classes -v $($data_folder)/results:/dataset/results -v $($data_folder_3D)/results_3D:/app/dataset/3D_data shawnpedro/silverfox /bin/bash
 shawnpedro/silverfox /bin/bash

