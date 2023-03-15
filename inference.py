import random
from pathlib import Path
from types import SimpleNamespace

import torch, wandb
from torch import nn
import numpy as np
from fastprogress import progress_bar



from cloud_diffusion.dataset import download_dataset, CloudDataset
from cloud_diffusion.ddpm import UNet2D, get_unet_params
from cloud_diffusion.utils import ddim_sampler, parse_args, set_seed, ls
from cloud_diffusion.wandb import log_images, save_model, to_video, vhtile

PROJECT_NAME = "ddpm_clouds"
DATASET_ARTIFACT = 'capecape/gtc/np_dataset:v1'
JOB_TYPE = "inference"
MODEL_ARTIFACT = "capecape/ddpm_clouds/esezp3jh_unet_small:v0"  # small model

config = SimpleNamespace(
    model_name="unet_small", # model name to save [unet_small, unet_big]
    sampler_steps=333, # number of sampler steps on the diffusion process
    num_frames=4, # number of frames to use as input,
    img_size=64, # image size to use
    num_random_experiments = 2, # we will perform inference multiple times on the same inputs
    seed=42,
    device="cuda:0",
    sampler="ddim",
    future_frames=10,  # number of future frames
    bs=16, # how many samples

)

set_seed(config.seed)

class Inference:

    def __init__(self, config):
        self.config = config

        # create a batch of data to use for inference
        self.prepare_data()
        
        # we default to ddim as it's faster and as good as ddpm
        self.sampler = ddim_sampler(config.sampler_steps)

        # create the Unet
        model_params = get_unet_params(config.model_name, config.num_frames)
        self.model = UNet2D.from_artifact(model_params, MODEL_ARTIFACT).to(config.device)

        self.model.eval()
    
    def prepare_data(self):
        "Generates a batch of data from the validation dataset"
        files = download_dataset(DATASET_ARTIFACT, PROJECT_NAME)

        self.valid_ds = CloudDataset(files=files[-3:], # 3 days of validation data 
                                num_frames=config.num_frames, img_size=config.img_size)
        self.idxs = random.choices(range(len(self.valid_ds) - config.future_frames), k=config.bs)  # select some samples
        self.batch = self.valid_ds[self.idxs].to(config.device)

    def sample_more(self, frames, n=1):
        "Autoregressive sampling, starting from `frames`. It is hardcoded to work with 3 frame inputs."
        for _ in progress_bar(range(n), total=n):
            new_frame = self.sampler(self.model, frames[:,-3:,...])
            frames = torch.cat([frames, new_frame.to(frames.device)], dim=1)
        return frames.cpu()

    def forecast(self):
        sequences = []
        for _ in range(config.num_random_experiments):
            frames = self.sample_more(self.batch, config.future_frames)
            sequences.append(frames)

        return sequences

    def log_to_wandb(self, sequences):
        table = wandb.Table(columns=["id", "gt", *[f"gen_{i}" for i in range(config.num_random_experiments)], "gt/gen"])
        
        for i, idx in enumerate(self.idxs):
            gt_vid = to_video(self.valid_ds[idx:idx+4+config.future_frames,0,...])
            pred_vids = [to_video(frames[i]) for frames in sequences]
            gt_gen = wandb.Image(vhtile(self.valid_ds[idx:idx+4+config.future_frames,0,...], *[frames[i] for frames in sequences]))
            table.add_data(idx, gt_vid, *pred_vids, gt_gen)
        
        wandb.log({f"gen_table_{config.future_frames}_random":table})

if __name__=="__main__":
    parse_args(config)
    with wandb.init(project=PROJECT_NAME, job_type=JOB_TYPE, 
                    config=config, tags=["ddpm", config.model_name]):
        infer = Inference(config)
        sequences = infer.forecast()
        infer.log_to_wandb(sequences)