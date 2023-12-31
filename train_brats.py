# -*- coding:utf-8 -*-
import nibabel as nib
from torchvision.transforms import Compose, Lambda
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from dataset_cubes import RandomCubeDataset
import torch
import os

# Setting CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

input_size = 16
depth_size = 16
num_channels = 64
num_res_blocks = 2
batchsize = 1
epochs = 10000
timesteps = 50
save_and_sample_every = 1000
with_condition = False  # Change to False if not needed
resume_weight = ""
in_channels = 1
out_channels = 1
channel_mult = ""
learn_sigma = False
class_cond = False
use_checkpoint = False
attention_resolutions = "16"
num_heads = 1
num_head_channels = -1
num_heads_upsample = -1
use_scale_shift_norm = False
dropout = 0
resblock_updown = False
use_fp16 = True
use_new_attention_order = False

model = create_model(
    image_size=input_size,
    num_channels=num_channels,
    num_res_blocks=num_res_blocks,
    channel_mult=channel_mult,
    learn_sigma=learn_sigma,
    in_channels=in_channels,
    out_channels=out_channels,
    class_cond=class_cond,
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
    with_condition=False,
    with_pairwised=False,
    apply_bce=False,
    lambda_bce=0.0,
).cuda()

dataset = RandomCubeDataset(size=(input_size, input_size, depth_size))

trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset,
    ema_decay=0.999,
    image_size=input_size,
    depth_size=depth_size,
    train_batch_size=batchsize,
    train_lr=2e-6,
    train_num_steps=epochs,
    gradient_accumulate_every=2,
    fp16=use_fp16,
    step_start_ema=2000,
    update_ema_every=10,
    save_and_sample_every=save_and_sample_every,
    results_folder="./results",
    with_condition=False,
    with_pairwised=False,
)

trainer.train()
