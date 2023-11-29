import os
import sys
import json

home_path = "/home/pedro/Desktop/Repos/SilverFox"
home_path = home_path if os.path.exists(os.path.join(home_path, "Dockerfile")) else ""

e040_path = "/home/j622s/Desktop/Silverfox/SilverFox-main"
e040_path = home_path if os.path.exists(home_path) else ""

docker_path = "/app/"
docker_path = docker_path if os.path.exists(os.path.join("/app", "Dockerfile")) else ""

if home_path:
    sys.path.append(home_path)
elif e040_path:
    sys.path.append(e040_path)
elif docker_path:
    sys.path.append(docker_path)
else:
    raise ValueError("Please specify the path of the project")

from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from datasets.dataset_alien import SimplyNumpyDataset4

# Setting CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# configuration of data
input_size = 512  # the size of image
depth_size = 0  # the size of classes


# configuration of training
batchsize = 4
epochs = 1e8
save_and_sample_every = 5500
resume_weight = ""
train_lr = 5e-5
step_start_ema = 1e4
gradient_accumulate_every = 4
update_ema_every = 10
ema_decay = 0.995

# configuration of network
num_channels = 192
num_res_blocks = 3
num_heads = 2
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
attention_resolutions = "8,16"
use_scale_shift_norm = False
resblock_updown = True
use_fp16 = True
use_new_attention_order = True

# configuration of diffusion process
timesteps = 500

data_folder = "/DATA/j622s/new_dataset_7/numpy_dataset_2D_128_classes"
results_folder = "/DATA/j622s/new_dataset_7/results_2D_512"

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

with open(os.path.join(results_folder, "config.txt"), "w") as f:
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
