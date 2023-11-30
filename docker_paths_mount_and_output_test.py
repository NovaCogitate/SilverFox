# this script will just check if the data exists, and we can load the dataset

import os
import sys
import json
import torch
import matplotlib.pyplot as plt

docker_path = "/app/"
docker_path = docker_path if os.path.exists(os.path.join("/app", "Dockerfile")) else ""

if docker_path:
    sys.path.append(docker_path)
else:
    raise ValueError("Please specify the path of the project")

from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from datasets.dataset_crosses import RandomXDataset, RandomCrossDataset
from datasets.dataset_alien import SimplyNumpyDataset4, Simply3D

# check if we have cuda available
assert torch.cuda.is_available(), "CUDA is not available"

# check if the results, results_2D and results_3D folder exists.
plot_folder, plot_folder_2D, plot_folder_3D = (
    "/app/results",
    "/app/results_2D",
    "/app/results_3D",
)

assert os.path.exists(plot_folder), "missing the results folder"
assert os.path.exists(plot_folder_2D), "missing the results 2D folder"
assert os.path.exists(plot_folder_3D), "missing the results 3D folder"


# check if the random 2D dataset works
INPUT_SIZE = 16
dataset_2D = RandomXDataset(size=(INPUT_SIZE, INPUT_SIZE))

# create a 2D dataloader
dataloader_2D = torch.utils.data.DataLoader(dataset_2D, batch_size=5, shuffle=True)

# save all the images in a new folder
for images in dataloader_2D:
    assert (
        len(images.shape) == 4
    ), "The 2D dataloader does not create samples with the correct shape. "
    b = images.shape[0]
    for i in range(b):
        image = images[i, 0, ...]
        image = image.cpu().numpy()
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(plot_folder_2D, f"test_sample_{i}.png"))
        plt.close()
    break

# check if the random 3D dataset works
dataset_3D = RandomCrossDataset(size=(INPUT_SIZE, INPUT_SIZE, INPUT_SIZE))

# create a dataloader 3D
dataloader_3D = torch.utils.data.DataLoader(dataset_3D, batch_size=5, shuffle=True)

for images in dataloader_3D:
    images = images.detach().cpu().numpy()
    b, c, _, w, z = images.shape
    for channel in range(c):
        for batch in range(b):
            _, axis = plt.subplots(1, 7, figsize=(25, 5))
            half_point_z = int(w / 2)
            half_point_x = int(z / 2)
            axis[0].imshow(
                images[batch][channel][half_point_x, :, :],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axis[1].imshow(
                images[batch][channel][:, half_point_x, :],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axis[2].imshow(
                images[batch][channel][:, :, 0],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axis[3].imshow(
                images[batch][channel][:, :, half_point_z // 2],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axis[4].imshow(
                images[batch][channel][:, :, half_point_z],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axis[5].imshow(
                images[batch][channel][:, :, half_point_z + half_point_z // 2],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axis[6].imshow(
                images[batch][channel][:, :, -1],
                vmin=-1,
                vmax=1,
                cmap="gray",
            )
            axis[0].set_title("X projection on the X axis (yz plane)")
            axis[1].set_title("X projection on the Y axis (xz plane)")
            axis[2].set_title("X projection on the Y axis (xy plane) - base")
            axis[3].set_title("X projection on the Z axis (xy plane) - 1/4 way through")
            axis[4].set_title("X projection on the Z axis (xy plane) - 2/4 way through")
            axis[5].set_title("X projection on the Z axis (xy plane) - 3/4 way through")
            axis[6].set_title("X projection on the Z axis (xy plane) - 4/4 way through")
            for ax in axis.flatten():
                ax.axis("off")
            plt.savefig(os.path.join(plot_folder, f"{batch}.png"))
            plt.close()
    break


# Setting CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# configuration of data
# configuration of data
input_size = 16  # the size of image
depth_size = 0  # the size of classes


# configuration of training
batchsize = 64
epochs = 20
save_and_sample_every = 10
resume_weight = ""
train_lr = 1e-3
step_start_ema = 2000
gradient_accumulate_every = 1
update_ema_every = 10
ema_decay = 0.995

# configuration of network
num_channels = 32
num_res_blocks = 2
num_heads = 1
num_head_channels = -1
num_heads_upsample = -1
dropout = 0.05
conv_resample = True
dims = 2
num_classes = 128

in_channels = 1
out_channels = 1

channel_mult = ""
learn_sigma = False
class_cond = True
with_cond = False
use_checkpoint = False
attention_resolutions = "8"
use_scale_shift_norm = False
resblock_updown = True
use_fp16 = True
use_new_attention_order = True

# configuration of diffusion process
timesteps = 10

# check if the real data 2D dataset works
dims = 2
DEFAULT_DATA_FOLDER = "/app/dataset/numpy_dataset_2D_128_classes"
DEFAULT_OUTPUT_FOLDER = f"/app/results/results_2D_{input_size}_nc:{num_channels}_nrb:{num_res_blocks}_nh:{num_heads}_att:{attention_resolutions}_lr:{train_lr}_timestep:{timesteps}"

dataset_real_2D = SimplyNumpyDataset4(
    path_to_dataset=DEFAULT_DATA_FOLDER,
    output_size=input_size,
)

# create a real 2D dataloader
dataloader_real_2D = torch.utils.data.DataLoader(
    dataset_real_2D, batch_size=5, shuffle=True
)

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

trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset_real_2D,
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
    results_folder=DEFAULT_OUTPUT_FOLDER,
    with_condition=with_cond,
    with_class_guided=class_cond,
)

trainer.train()

# check if the real data 3D dataset works
dims = 3
DEFAULT_DATA_FOLDER = "/app/data_folder/numpy_dataset_2D_128_classes"
DEFAULT_OUTPUT_FOLDER = f"/app/results/results_3D_{input_size}_nc:{num_channels}_nrb:{num_res_blocks}_nh:{num_heads}_att:{attention_resolutions}_lr:{train_lr}_timestep:{timesteps}"

# configuration of data
input_size = 16  # the size of image
depth_size = 16  # the size of classes

# configuration of training
batchsize = 8
epochs = 20000
save_and_sample_every = 500
resume_weight = ""
train_lr = 5e-4
step_start_ema = 2000
gradient_accumulate_every = 1
update_ema_every = 10
ema_decay = 0.995

# configuration of network
num_channels = 64
num_res_blocks = 3
num_heads = 1
num_head_channels = -1
num_heads_upsample = -1
dropout = 0.05
conv_resample = True
dims = 3
num_classes = None

in_channels = 1
out_channels = 1

channel_mult = ""
learn_sigma = False
class_cond = False
use_checkpoint = False
attention_resolutions = "16,8"
use_scale_shift_norm = False
resblock_updown = True
use_fp16 = True
use_new_attention_order = True

# configuration of diffusion process
timesteps = 10

dataset_real_3D = Simply3D(
    DEFAULT_DATA_FOLDER,
    output_size=input_size,
    depth_size=depth_size,
)

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

trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset_real_3D,
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
    results_folder=DEFAULT_OUTPUT_FOLDER,
    with_condition=class_cond,
)

trainer.train()
