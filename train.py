# Modified from DiT official Repo.
# Training Diffusion Tuning with a single GPU.

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from download import find_model

from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# List of trainable parameters
#################################################################################
#                             Difffit baseline (PEFT)                           #
#################################################################################
trainable_names = ["bias","norm","gamma","y_embed"]
blacklist_names = ["copy", "ema"]

def brute_force_match(name):
    for blacklist_name in blacklist_names:
        if blacklist_name in name:
            return False
    for trainable_name in trainable_names:
        if trainable_name in name:
            return True
    return False

def diff_fit(model):# Step 1: Freeze all params

    for name, param in model.named_parameters():
        param.requires_grad = False# Step 2: Unfreeze specific paramsfor name, param in model.named_parameters():# unfreeze specific parameters with nameif match(name, trainable_names):param.requires_grad = True# Step 3: Fine-tuningtrain(model, data, epochs
        # print(name)
    for name, param in model.named_parameters(): # unfreeze specific parameters with name 
        if brute_force_match(name): 
            param.requires_grad = True
            # print("chong",name,param)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting, seed={seed}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    print("device: ",device)
    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_gamma=args.difffit,
        device = device,
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor

    ## fine-tuning
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if args.difffit: # for reproduce difffit.
        print("run with difffit...")
        diff_fit(model)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable Parameters: {sum(p.numel() for _, p in model.named_parameters() if p.requires_grad):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    
    batch_size = int(args.global_batch_size)
    ## LWF setting
    if args.y_embed_mode == "LWF":
        lwf_dataset = ImageFolder("pretrained_images", transform=transform)
        lwf_batch_size = int(args.global_batch_size * args.LWF_ratio)
        batch_size = int(args.global_batch_size - lwf_batch_size)
        print(f"LWF mode running: LWF batch : True batch{lwf_batch_size}:{batch_size}")
        lwf_loader = DataLoader(
            lwf_dataset,
            batch_size=lwf_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"LWF_Dataset contains {len(lwf_dataset):,} images ({args.lwf_step})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    if args.LWF_prob_scale < 0:
        args.LWF_prob_scale = args.prob_scale

    probs = torch.Tensor([i**args.prob_scale for i in range (1,1001)]) # sampling probs
    rev_probs = torch.Tensor([i**args.LWF_prob_scale for i in range (1000,0,-1)]) 
    probs = probs / probs.sum()
    rev_probs = rev_probs / rev_probs.sum()

    categorical_dist = torch.distributions.categorical.Categorical(probs=probs)
    rev_categorical_dist = torch.distributions.categorical.Categorical(probs=rev_probs)
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_s = 0
    start_time = time()
    print("args.P2",args.p2)
    print("args.min_snr",args.min_snr)

    if args.t_prob == 'categorical':
        logger.info(f"categorical with scale, {args.prob_scale}, rev prob: {args.LWF_prob_scale}")

    if args.y_embed_mode == "LWF":
        lwf_iter = iter(lwf_loader)
    logger.info(f"Training for {args.epochs} epochs... with  lr {args.lr}")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...with train mode:{args.y_embed_mode}")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if args.y_embed_mode == "LWF":
                try:
                    x_s, y_s = next(lwf_iter)
                except StopIteration:
                    lwf_iter = iter(lwf_loader)
                    x_s, y_s = next(lwf_iter)
                x_s = x_s.to(device)
                y_s = y_s.to(device)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                if args.y_embed_mode == "LWF":
                    x_s = vae.encode(x_s).latent_dist.sample().mul_(0.18215)
    
            if args.t_prob == 'categorical':
                t = categorical_dist.sample((x.shape[0],)).to(device)
                if args.y_embed_mode == "LWF":
                    t_s = rev_categorical_dist.sample((x_s.shape[0],)).to(device)
            elif args.t_prob == 'uniform':
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                if args.y_embed_mode == "LWF":
                    t_s = torch.randint(0, diffusion.num_timesteps, (x_s.shape[0],), device=device)
            else:
                raise NotImplementedError
    

            if args.y_embed_mode == "default" or args.y_embed_mode == "scratch":
                y_embed_split = None
            elif args.y_embed_mode == "LWF":
                y_embed_split = [batch_size, lwf_batch_size]
                x = torch.cat([x, x_s], dim=0)
                y = torch.cat([y, y_s], dim=0)
                t = torch.cat([t, t_s], dim=0)
            model_kwargs = dict(y=y, y_embed_split=y_embed_split)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs, P2_weighting=args.p2, MIN_SNR=args.min_snr)
            loss = loss_dict["loss"]
            if args.y_embed_mode == "LWF":
                loss_t, loss_s = torch.split(loss, [batch_size, lwf_batch_size], dim=0)
                loss_t = loss_t.mean()
                loss_s = loss_s.mean()
                loss = loss_t + loss_s * args.LWF_weight
                running_loss += loss_t.item()
                running_loss_s += loss_s.item()
            else:
                loss = loss.mean()
                running_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                # torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                # avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                if args.y_embed_mode == "LWF":
                    avg_loss_s = torch.tensor(running_loss_s / log_steps, device=device)
                    logger.info(f"(step={train_steps:07d}) Train Loss_s: {avg_loss_s:.4f}")
                    running_loss_s = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            if train_steps >= args.max_steps:
                print(f"finished at {args.max_steps}")
                logger.info("Done!")
                return 0
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=24000)
    parser.add_argument("--max-steps", type=int, default=24000)
    
    parser.add_argument("--y-embed-mode", type=str, choices=["scratch", "LWF", "default"], default="default")
    parser.add_argument("--LWF-ratio", type=float, default=0.2)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--t_prob", type=str, choices=["uniform","categorical"])
    parser.add_argument("--prob-scale", type=float, default=1.0)
    parser.add_argument("--mask-trick", action='store_true', help="Use DiffFit instead of DiT")

    parser.add_argument("--difffit", action='store_true', help="Use DiffFit instead of DiT")
    parser.add_argument("--difffit-depth",default=14,type=int, help="Depth of DiffFit model")
    parser.add_argument("--p2", action='store_true', help="Use P2 weighting")
    parser.add_argument("--LWF-weight", type=float, default=0.1)
    parser.add_argument("--LWF-prob-scale", type=float, default=-1)
    parser.add_argument("--LWF-uncond-only", action='store_true', help="Use only unconditional LWF")
    parser.add_argument("--min-snr", type=float, default=-1)
    args = parser.parse_args()
    main(args)
