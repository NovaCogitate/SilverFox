import sys
import json
import os

# all the possible working paths.
home_pc_path = "/home/pedro/Desktop/Repos/SilverFox"
laptop_path = "/home/pedro/Desktop/Local_re_Work/SilverFox"
e040_path = "/home/j622s/Desktop/SilverFox"
odcf_path = "/home/j622s/SilverFox"
docker_path = "/app/"

# go to a random file and check if the location exists. In this case the docker file.
home_exists = os.path.exists(os.path.join(home_pc_path, "Dockerfile"))
laptop_exists = os.path.exists(os.path.join(laptop_path, "Dockerfile"))
e040_exists = os.path.exists(os.path.join(e040_path, "Dockerfile"))
odcf_exists = os.path.exists(os.path.join(odcf_path, "Dockerfile"))
docker_exists = os.path.exists(os.path.join(docker_path, "Dockerfile"))

total_true_conditions = sum(
    [home_exists, laptop_exists, e040_exists, odcf_exists, docker_exists]
)
if total_true_conditions > 1:
    raise ValueError("More than one path exists!")
elif total_true_conditions == 0:
    raise FileNotFoundError("Path to SilverFox-main not found!")
else:
    pass

home_data_path = "/home/pedro/Desktop/Repos/SilverFox_data/numpy_dataset_2D_128_classes"
laptop_data_path = (
    "/home/pedro/Desktop/Local_re_Work/SilverFox_data/numpy_dataset_2D_128_classes"
)
e040_data_path = "/DATA/j622s/new_dataset_7/numpy_dataset_2D_128_classes"
odcf_data_path = "/dkfz/cluster/gpu/data/OE0094/j622s/numpy_dataset_2D_128_classes"
docker_data_path = "/app/data_folder/numpy_dataset_2D_128_classes"

home_results_path = "/home/pedro/Desktop/Repos/results/"
laptop_results_path = "/home/pedro/Desktop/Local_re_Work/fox_output"
e040_results_path = "/DATA/j622s/new_dataset_7/output/SilverResults"
odcf_results_path = "/dkfz/cluster/gpu/data/OE0094/j622s/SilverResults"
docker_results_path = "app/results/"

# Setting CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# configuration of data
input_size = 256  # the size of image
depth_size = 0  # the size of classes

# configuration of training
batchsize = 8
epochs = 1e7
save_and_sample_every = 2500
resume_weight = ""
train_lr = 1e-4
step_start_ema = 1e4
gradient_accumulate_every = 2
update_ema_every = 10
ema_decay = 0.995

# configuration of network
num_channels = 96
num_res_blocks = 2
num_heads = 3
num_head_channels = -1
num_heads_upsample = -1
dropout = 0.05
conv_resample = True
dims = 2
num_classes = 128
resblock_updown = True
use_new_attention_order = True

in_channels = 1
out_channels = 1

channel_mult = ""
learn_sigma = False
class_cond = True
with_cond = False
use_checkpoint = False
attention_resolutions = "16"
use_scale_shift_norm = False
use_fp16 = True
timesteps = 500
result_name = f"results_2D_{input_size}_nc:{num_channels}_att:{attention_resolutions}_lr:{train_lr}_timestep:{timesteps}_nh:{num_heads}_nrb:{num_res_blocks}"

if home_exists:
    data_folder = home_data_path
    results_folder = home_results_path + "/" + result_name
    sys.path.append(home_pc_path)
elif laptop_exists:
    data_folder = laptop_data_path
    results_folder = laptop_results_path + "/" + result_name
    sys.path.append(laptop_path)
elif e040_exists:
    data_folder = e040_data_path
    results_folder = e040_results_path + "/" + result_name
    sys.path.append(e040_path)
elif odcf_exists:
    data_folder = odcf_data_path
    results_folder = odcf_results_path + "/" + result_name
    sys.path.append(odcf_path)
elif docker_exists:
    data_folder = docker_data_path
    results_folder = docker_results_path + "/" + result_name
    sys.path.append(docker_path)
else:
    raise FileNotFoundError("Path to SilverFox-main not found!")
    raise FileNotFoundError("Path to SilverFox-main not found!")

from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from datasets.dataset_alien import SimplyNumpyDataset4

# the configs above
config = {
    "batchsize": batchsize,
    "epochs": epochs,
    "save_and_sample_every": save_and_sample_every,
    "resume_weight": resume_weight,
    "train_lr": train_lr,
    "step_start_ema": step_start_ema,
    "gradient_accumulate_every": gradient_accumulate_every,
    "update_ema_every": update_ema_every,
    "ema_decay": ema_decay,
    "num_channels": num_channels,
    "num_res_blocks": num_res_blocks,
    "num_heads": num_heads,
    "num_head_channels": num_head_channels,
    "num_heads_upsample": num_heads_upsample,
    "dropout": dropout,
    "conv_resample": conv_resample,
    "dims": dims,
    "num_classes": num_classes,
    "in_channels": in_channels,
    "out_channels": out_channels,
    "channel_mult": channel_mult,
    "learn_sigma": learn_sigma,
    "class_cond": class_cond,
    "use_checkpoint": use_checkpoint,
    "attention_resolutions": attention_resolutions,
    "use_scale_shift_norm": use_scale_shift_norm,
    "resblock_updown": resblock_updown,
    "use_fp16": use_fp16,
    "use_new_attention_order": use_new_attention_order,
    "timesteps": timesteps,
    "input_size": input_size,
    "depth_size": depth_size,
}

# save the config
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

with open(os.path.join(results_folder, "config.txt"), "w", encoding="utf-8") as f:
    json.dump(config, f)

model = create_model(
    image_size=input_size,
    num_channels=num_channels,
    num_res_blocks=num_res_blocks,
    channel_mult=channel_mult,
    learn_sigma=learn_sigma,
    in_channels=in_channels,
    out_channels=out_channels,
    class_cond=class_cond,
    conv_resample=conv_resample,
    dims=dims,
    use_checkpoint=use_checkpoint,
    attention_resolutions=attention_resolutions,
    num_heads=num_heads,
    num_head_channels=num_head_channels,
    num_heads_upsample=num_heads_upsample,
    use_scale_shift_norm=use_scale_shift_norm,
    dropout=dropout,
    resblock_updown=resblock_updown,
    use_fp16=use_fp16,
    use_new_attention_order=use_new_attention_order,
).cuda()

diffusion = GaussianDiffusion(
    denoise_fn=model,
    image_size=input_size,
    depth_size=depth_size,
    channels=in_channels,
    timesteps=timesteps,
    loss_type="l1",
    betas=None,
    with_condition=with_cond,
    with_pairwised=False,
    apply_bce=False,
    lambda_bce=0.0,
).cuda()

dataset = SimplyNumpyDataset4(
    path_to_dataset=data_folder,
    output_size=input_size,
)

trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset,
    ema_decay=ema_decay,
    depth_size=depth_size,
    train_batch_size=batchsize,
    train_lr=train_lr,
    train_num_steps=epochs,
    gradient_accumulate_every=gradient_accumulate_every,
    fp16=use_fp16,
    step_start_ema=step_start_ema,
    update_ema_every=update_ema_every,
    save_and_sample_every=save_and_sample_every,
    results_folder=results_folder,
    with_condition=with_cond,
    with_class_guided=class_cond,
)

trainer.train()
