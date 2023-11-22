import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import random

sys.path.append("/home/pedro/Desktop/Repos/SilverFox/")
from datasets.dataset_crosses import RandomCrossDataset, draw_x, RandomXDataset

# load the model
import nibabel as nib
from torchvision.transforms import Compose, Lambda

import sys
import json


from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from datasets.dataset_crosses import RandomCrossDataset
import torch
import os

path_to_loading_folder = "/home/pedro/Desktop/Repos/SilverFox_toy_back_ups/results_3D"
path_to_config_file = os.path.join(path_to_loading_folder, "config.txt")
assert os.path.exists(path_to_config_file), "The config file does not exist"

with open(os.path.join(path_to_config_file), "r") as f:
    dictionary = json.load(f)

# load the parameters
input_size = dictionary["input_size"]
depth_size = dictionary["depth_size"]
num_channels = dictionary["num_channels"]
num_res_blocks = dictionary["num_res_blocks"]
channel_mult = dictionary["channel_mult"]
learn_sigma = dictionary["learn_sigma"]
in_channels = dictionary["in_channels"]
out_channels = dictionary["out_channels"]
class_cond = dictionary["class_cond"]
use_checkpoint = dictionary["use_checkpoint"]
attention_resolutions = dictionary["attention_resolutions"]
num_heads = dictionary["num_heads"]
num_head_channels = dictionary["num_head_channels"]
num_heads_upsample = dictionary["num_heads_upsample"]
use_scale_shift_norm = dictionary["use_scale_shift_norm"]
dropout = dictionary["dropout"]
resblock_updown = dictionary["resblock_updown"]
# use_fp16 = dictionary["use_fp16"]
dims = dictionary["dims"]
conv_resample = dictionary["conv_resample"]
use_fp16 = False
use_new_attention_order = dictionary["use_new_attention_order"]
batchsize = dictionary["batchsize"]
train_lr = dictionary["train_lr"]
epochs = dictionary["epochs"]
gradient_accumulate_every = dictionary["gradient_accumulate_every"]
ema_decay = dictionary["ema_decay"]
step_start_ema = dictionary["step_start_ema"]
update_ema_every = dictionary["update_ema_every"]
save_and_sample_every = dictionary["save_and_sample_every"]
# results_folder = dictionary["results_folder"]
results_folder = path_to_loading_folder
timesteps = dictionary["timesteps"]


# load the model
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
    with_condition=class_cond,
    with_pairwised=False,
    apply_bce=False,
    lambda_bce=0.0,
).cuda()

dataset = RandomCrossDataset(size=(input_size, input_size, depth_size), half=True)

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
    with_condition=class_cond,
)

trainer.load(milestone=10)

trainer.generate_samples(no_of_samples=10, batch_size=2, milestone=10)
