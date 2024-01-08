import copy
import torch
import datetime
import time
import os
import warnings

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn

from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from apex import amp

    APEX_AVAILABLE = True
    print("APEX: ON")
except ModuleNotFoundError:
    APEX_AVAILABLE = False
    print("APEX: OFF")


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels=1,
        timesteps=250,
        loss_type="l1",
        betas=None,
        with_condition=False,
        with_class_guidance=False,
        with_pairwised=False,
        apply_bce=False,
        lambda_bce=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.with_class_guidance = with_class_guidance
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce

        if exists(betas):
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )
        self.register_buffer(
            "posterior_mean_coef3",
            to_torch(
                1.0
                - (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def q_mean_variance(self, x_start, t, c=None):
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.0
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.0
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            + extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c=None, y=None):
        if self.with_condition:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([x, c], 1), t)
            )
        elif self.with_class_guidance:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t, y=c)
            )
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t, c=c
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        condition_tensors=None,
        clip_denoised=True,
        repeat_noise=False,
        y=None,
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised, y=y
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors=None, y=None):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            if self.with_condition:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t, condition_tensors=condition_tensors)
            elif self.with_class_guidance:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t, y=y)
            else:
                img = self.p_sample(
                    img, torch.full((b,), i, device=device, dtype=torch.long)
                )

        return img

    @torch.no_grad()
    def sample(self, batch_size=2, condition_tensors=None, y=None):
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels

        if depth_size == 0:
            return self.p_sample_loop(
                (batch_size, channels, image_size, image_size),
                condition_tensors=condition_tensors,
                y=y,
            )
        elif depth_size > 0:
            return self.p_sample_loop(
                (batch_size, channels, depth_size, image_size, image_size),
                condition_tensors=condition_tensors,
                y=y,
            )
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )
        return img

    def q_sample(self, x_start, t, noise=None, c=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.0
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            + x_hat
        )

    def p_losses(self, x_start, t, y=None, condition_tensors=None, noise=None):
        # b, c, h, w, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.with_condition:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], 1), t)
        elif self.with_class_guidance:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t, y=y)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == "l1":
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise)
        elif self.loss_type == "hybrid":
            loss1 = (noise - x_recon).abs().mean()
            #             loss2 = F.mse_loss(x_recon, noise)
            loss2 = F.binary_cross_entropy_with_logits(x_recon, noise)
            loss = loss1 * 100 + loss2
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        """This is a forwards step of the diffusion model.

        Args:
            x (_type_): _description_
            condition_tensors (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if len(x.shape) == 5:
            b, c, d, h, w = x.shape
        else:
            b, c, h, w = x.shape

        (
            device,
            img_size,
        ) = (
            x.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}, but now h={h} and w={w}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)


# trainer class


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        depth_size=128,
        train_batch_size=1,
        train_lr=2e-6,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder="./results",
        with_condition=False,
        with_class_guided=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.with_class_guided = with_class_guided
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
        )
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition

        self.step = 0

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize(
                [self.model, self.ema_model], self.opt, opt_level="O1"
            )
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        os.makedirs(f"{results_folder}/model", exist_ok=True)

        self.log_dir = self.create_log_dir()

        self.reset_parameters()

    def create_log_dir(self):
        now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
        log_dir = os.path.join(".logs", now)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data_to_save = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        torch.save(
            data_to_save, str(self.results_folder / f"model/model-{milestone}.pt")
        )

    def load(self, milestone):
        data_to_load = torch.load(
            str(self.results_folder / f"model/model-{milestone}.pt")
        )
        self.step = data_to_load["step"]
        self.model.load_state_dict(data_to_load["model"])
        self.ema_model.load_state_dict(data_to_load["ema"])

    def train(self):
        loss_values = []
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps:
            accumulated_loss = []
            for _ in range(self.gradient_accumulate_every):
                if self.with_condition:
                    data_iter = next(self.dl)
                    input_tensors = data_iter[0].cuda()
                    # target_tensors = data_iter["segmentation"].cuda()
                    class_condition = torch.tensor(data_iter[1]).cuda()
                    loss = self.model(input_tensors, condition_tensors=class_condition)
                elif self.with_class_guided:
                    data_item, class_item = next(self.dl)
                    input_tensors = data_item.cuda()
                    class_condition = torch.tensor(class_item).cuda()
                    loss = self.model(input_tensors, y=class_condition)
                else:
                    data_iter = next(self.dl).cuda()
                    loss = self.model(data_iter)

                loss = loss.sum() / self.batch_size
                print(f"{self.step}: {loss.item()}")
                backwards(loss / self.gradient_accumulate_every, self.opt)
                accumulated_loss.append(loss.item())

            # Record here
            average_loss = np.mean(accumulated_loss)
            loss_values.append(average_loss)
            end_time = time.time()
            print("training_loss", average_loss, self.step)
            # self.writer.add_scalar("training_loss", average_loss, self.step)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                # save the losses and the rolling average losses
                np.save(
                    f"{self.results_folder}/model/loss_values.npy",
                    np.array(loss_values),
                )
                # draw a plot
                # calculate the rolling avarages
                rolling_average = []

                for i in range(len(loss_values)):
                    if i < 10:
                        rolling_average.append(loss_values[i])
                    else:
                        rolling_average.append(np.mean(loss_values[i - 10 : i]))

                plt.plot(rolling_average)
                plt.plot(loss_values)
                plt.loglog()

                plt.title("Loss values")
                plt.xlabel("Steps")
                plt.ylabel("Loss")

                plt.savefig(f"{self.results_folder}/model/loss_values.png")

                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(1, self.batch_size)

                if self.with_condition:
                    all_images_list = list(
                        map(
                            lambda n: self.ema_model.sample(
                                batch_size=n,
                                condition_tensors=self.ds.sample_conditions(
                                    batch_size=n
                                ),
                            ),
                            batches,
                        )
                    )
                    all_images = torch.cat(all_images_list, dim=0)
                elif self.with_class_guided:
                    classes = list(range(0, 129, 16))
                    no_of_classes = len(classes)
                    all_images_list = list(
                        map(
                            lambda n: self.ema_model.sample(
                                batch_size=n,
                                y=torch.tensor(classes),
                            ),
                            [no_of_classes],
                        )
                    )
                    all_images = torch.cat(all_images_list, dim=0)
                else:
                    all_images_list = list(
                        map(lambda n: self.ema_model.sample(batch_size=n), batches)
                    )
                    all_images = torch.cat(all_images_list, dim=0)

                output = all_images[:, 0, ...]
                output = output.cpu().numpy()

                np.save(
                    f"{self.results_folder}/model/output-{milestone}.npy",
                    output,
                )
                self.save(milestone)

                if len(output.shape) == 4:
                    b, _, w, z = output.shape
                    for batch in range(b):
                        _, axis = plt.subplots(1, 7, figsize=(25, 5))
                        half_point_z = int(w / 2)
                        half_point_x = int(z / 2)

                        axis[0].imshow(
                            output[batch][half_point_x, :, :],
                            vmin=-1,
                            vmax=1,
                            cmap="gray",
                        )
                        axis[1].imshow(
                            output[batch][:, half_point_x, :],
                            vmin=-1,
                            vmax=1,
                            cmap="gray",
                        )

                        axis[2].imshow(
                            output[batch][:, :, 0],
                            vmin=-1,
                            vmax=1,
                            cmap="gray",
                        )

                        axis[3].imshow(
                            output[batch][:, :, half_point_z // 2],
                            vmin=-1,
                            vmax=1,
                            cmap="gray",
                        )
                        axis[4].imshow(
                            output[batch][:, :, half_point_z],
                            vmin=-1,
                            vmax=1,
                            cmap="gray",
                        )
                        axis[5].imshow(
                            output[batch][:, :, half_point_z + half_point_z // 2],
                            vmin=-1,
                            vmax=1,
                            cmap="gray",
                        )

                        axis[6].imshow(
                            output[batch][:, :, -1],
                            vmin=-1,
                            vmax=1,
                            cmap="gray",
                        )

                        axis[0].set_title("X projection on the X axis (yz plane)")
                        axis[1].set_title("X projection on the Y axis (xz plane)")
                        axis[2].set_title(
                            "X projection on the Y axis (xy plane) - base"
                        )
                        axis[3].set_title(
                            "X projection on the Z axis (xy plane) - 1/4 way through"
                        )
                        axis[4].set_title(
                            "X projection on the Z axis (xy plane) - 2/4 way through"
                        )
                        axis[5].set_title(
                            "X projection on the Z axis (xy plane) - 3/4 way through"
                        )
                        axis[6].set_title(
                            "X projection on the Z axis (xy plane) - 4/4 way through"
                        )

                        for ax in axis.flatten():
                            ax.axis("off")

                        plt.savefig(
                            f"{self.results_folder}/model/output-{milestone}-{batch}.png"
                        )
                        plt.close()

                elif len(output.shape) == 3 and self.with_class_guided:
                    b, _, w = output.shape
                    if b > 1:
                        _, axis = plt.subplots(1, b, figsize=(15, 5))

                        for batch, ax in enumerate(axis.flatten()):
                            ax.axis("off")
                            ax.imshow(output[batch, :, :], vmin=-1, vmax=1, cmap="gray")
                    else:
                        _, axis = plt.subplots(1, 1, figsize=(15, 5))
                        axis.axis("off")
                        axis.imshow(output[0, :, :], vmin=-1, vmax=1, cmap="gray")
                        axis.set_title("x-y plane")

                    plt.savefig(
                        f"{self.results_folder}/model/output-{milestone}-{b}.png"
                    )
                    plt.close()

                elif len(output.shape) == 3:
                    b, _, w = output.shape
                    if b > 1:
                        _, axis = plt.subplots(1, b, figsize=(15, 5))

                        for batch, ax in enumerate(axis.flatten()):
                            ax.axis("off")
                            ax[batch].imshow(
                                output[batch, :, :], vmin=-1, vmax=1, cmap="gray"
                            )
                    else:
                        _, axis = plt.subplots(1, 1, figsize=(15, 5))
                        axis.axis("off")
                        axis.imshow(output[0, :, :], vmin=-1, vmax=1, cmap="gray")
                        axis.set_title("x-y plane")

                    plt.savefig(
                        f"{self.results_folder}/model/output-{milestone}-{b}.png"
                    )
                    plt.close()

            self.step += 1

        print("training completed")
        end_time = time.time()
        execution_time = (end_time - start_time) / 3600
        print(f"Execution time (hour): {execution_time}")

    def generate_and_return_samples(
        self,
        no_of_samples,
        batch_size,
        conditioning_samples=None,
        y=torch.tensor(list(range(0, 128, 16))),
    ):
        # number of batches need to be calculated
        batches = num_to_groups(no_of_samples, batch_size)

        # splitting the conditional tensors:
        if conditioning_samples:
            split_tensors = []
            start = 0
            for length in batches:
                end = start + length
                split_tensors.append(conditioning_samples[start:end])
                start = end

        # splitting the conditional classes
        if isinstance(y, torch.Tensor):
            split_classes = []
            start = 0
            for length in batches:
                end = start + length
                split_classes.append(y[start:end])
                start = end

        if self.with_condition:
            all_images_list = []
            for batch, split_tensor in zip(batches, split_tensors):
                all_images_list.append(
                    self.ema_model.sample(
                        batch_size=batch,
                        condition_tensors=split_tensor,
                    )
                )
            all_images = torch.cat(all_images_list, dim=0)

        elif self.with_class_guided:
            all_images_list = []
            for batch, split_sample in zip(batches, split_classes):
                all_images_list.append(
                    self.ema_model.sample(
                        batch_size=batch,
                        y=split_sample,
                    )
                )

            all_images = torch.cat(all_images_list, dim=0)

        else:
            all_images_list = list(
                map(lambda n: self.ema_model.sample(batch_size=n), batches)
            )
            all_images = torch.cat(all_images_list, dim=0)

        return all_images

    def generate_samples(
        self, no_of_samples, batch_size, conditioning_samples=None, milestone=0
    ):
        # number of batches need to be calculated
        batches = num_to_groups(no_of_samples, batch_size)

        # Splitting the array
        split_tensors = []
        start = 0
        for length in batches:
            end = start + length
            split_tensors.append(conditioning_samples[start:end])
            start = end

        if self.with_condition:
            all_images_list = list(
                map(
                    lambda batch: self.ema_model.sample(
                        batch_size=batch,
                        condition_tensors=split_tensors,
                    ),
                    batches,
                )
            )
            all_images = torch.cat(all_images_list, dim=0)
        elif self.with_class_guided:
            all_images_list = list(
                map(
                    lambda n: self.ema_model.sample(batch_size=n),
                    batches,
                    y=torch.tensor(list(range(0, 128, 16))),
                )
            )
            all_images = torch.cat(all_images_list, dim=0)

        else:
            all_images_list = list(
                map(lambda n: self.ema_model.sample(batch_size=n), batches)
            )
            all_images = torch.cat(all_images_list, dim=0)

        plot_folder = os.path.join(self.results_folder, f"plots_{milestone}")
        os.makedirs(plot_folder, exist_ok=True)

        if len(all_images.shape) == 5:
            all_images = all_images.detach().cpu().numpy()
            b, c, _, w, z = all_images.shape
            for channel in range(c):
                for batch in range(b):
                    _, axis = plt.subplots(1, 7, figsize=(25, 5))
                    half_point_z = int(w / 2)
                    half_point_x = int(z / 2)

                    axis[0].imshow(
                        all_images[batch][channel][half_point_x, :, :],
                        vmin=-1,
                        vmax=1,
                        cmap="gray",
                    )
                    axis[1].imshow(
                        all_images[batch][channel][:, half_point_x, :],
                        vmin=-1,
                        vmax=1,
                        cmap="gray",
                    )

                    axis[2].imshow(
                        all_images[batch][channel][:, :, 0],
                        vmin=-1,
                        vmax=1,
                        cmap="gray",
                    )

                    axis[3].imshow(
                        all_images[batch][channel][:, :, half_point_z // 2],
                        vmin=-1,
                        vmax=1,
                        cmap="gray",
                    )
                    axis[4].imshow(
                        all_images[batch][channel][:, :, half_point_z],
                        vmin=-1,
                        vmax=1,
                        cmap="gray",
                    )
                    axis[5].imshow(
                        all_images[batch][channel][
                            :, :, half_point_z + half_point_z // 2
                        ],
                        vmin=-1,
                        vmax=1,
                        cmap="gray",
                    )

                    axis[6].imshow(
                        all_images[batch][channel][:, :, -1],
                        vmin=-1,
                        vmax=1,
                        cmap="gray",
                    )

                    axis[0].set_title("X projection on the X axis (yz plane)")
                    axis[1].set_title("X projection on the Y axis (xz plane)")
                    axis[2].set_title("X projection on the Y axis (xy plane) - base")
                    axis[3].set_title(
                        "X projection on the Z axis (xy plane) - 1/4 way through"
                    )
                    axis[4].set_title(
                        "X projection on the Z axis (xy plane) - 2/4 way through"
                    )
                    axis[5].set_title(
                        "X projection on the Z axis (xy plane) - 3/4 way through"
                    )
                    axis[6].set_title(
                        "X projection on the Z axis (xy plane) - 4/4 way through"
                    )

                    for ax in axis.flatten():
                        ax.axis("off")

                    plt.savefig(os.path.join(plot_folder, f"{batch}.png"))
                    np.save(
                        os.path.join(plot_folder, f"{batch}.npy"), all_images[batch]
                    )
                    plt.close()

        elif len(all_images.shape) == 4:
            # save all the images in a new folder
            for i in range(all_images.shape[0]):
                image = all_images[i, 0, ...]
                image = image.cpu().numpy()
                plt.imshow(image, cmap="gray")
                plt.axis("off")
                plt.savefig(os.path.join(plot_folder, f"sample_{i}.png"))
                plt.close()
                np.save(os.path.join(plot_folder, f"sample_{i}.npy"), image)

                # save the numpy as well
                np.save(os.path.join(plot_folder, f"sample_{i}.npy"), image)

        else:
            raise NotImplementedError()
