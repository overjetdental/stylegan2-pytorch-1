import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator
from swagan import Generator as SWAGenerator
from bicubic import BicubicDownSample

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--latent", type=int, default=512, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--n_mlp", type=int, default=8, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "--swagan",
        action="store_true",
        help="use swagan",
    )
    parser.add_argument(
        "--low_res_size",
        type=int,
        default=None,
        help="this is using the loss on low resolution images ",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = max(args.size, 256)

    # sets the lowres downsize
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),  # TODO change this to resize and pad
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Downsampling functions
    if args.low_res_size:
        downsampler_512_256 = BicubicDownSample(2)
        downsampler_512_lr = BicubicDownSample(512 // args.low_res_size)
        upsampler_lr_256 = torch.nn.Upsample(256)

    to_256_lpips = torch.nn.Upsample(256)
    # Initialize Stylegan2
    if args.swagan:
        g_ema = SWAGenerator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier)
    else:
        g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier)
    # Load checkpoint
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

    imgs = []
    if os.path.isdir(args.files[0]):
        img_files = [os.path.join(args.files[0], fps) for fps in os.listdir(args.files[0])]
    else:
        img_files = args.files

    for n, imgfile in enumerate(img_files):
        print('Filename: {}'.format(imgfile))
        img = transform(Image.open(imgfile).convert("RGB"))

        imgs = [img]
        imgs = torch.stack(imgs, 0).to(device)

        # Downsample images to get the new gt.
        if args.low_res_size:
            imgs = downsampler_512_lr(imgs)

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            # need to change the size so we are comparing low res examples to low res examples
            if args.low_res_size:
                img_gen = downsampler_512_lr(img_gen)

            batch, channel, height, width = img_gen.shape

            """
            if height > 256:  # TODO: understand if this is already a square
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])
            """
            #
            #if args.low_res_size:
            #    p_loss = percept(upsampler_lr_256(img_gen), upsampler_lr_256(imgs)).sum()
            #else:

            p_loss = percept(to_256_lpips(img_gen), to_256_lpips(imgs)).sum()

            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

        filename = os.path.splitext(os.path.basename(imgfile))[0] + ".pt"
        img_ar = make_image(img_gen)
        result_file = {}
        # for i, input_name in enumerate(imgfile):
        # noise_single = []
        # for noise in noises:
        #    noise_single.append(noise)

        result_file[imgfile] = {
            "img": img_gen,
            "latent": latent_in,
            # "noise": noise_single,
            "perceptual_loss": p_loss,
            "mse_loss": mse_loss,
            "noise_loss": n_loss,
            "loss": loss,
        }
        img_name = os.path.splitext(os.path.basename(imgfile))[0] + "-project.png"
        pil_img = Image.fromarray(img_ar[0])
        pil_img.save(os.path.join(os.getcwd(), 'projection_examples', 'outputs', img_name))

        torch.save(result_file, os.path.join(os.getcwd(), 'projection_examples', 'outputs', filename))
