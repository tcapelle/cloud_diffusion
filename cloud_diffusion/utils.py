import random, argparse
from pathlib import Path

import wandb
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data.dataloader import default_collate

from fastprogress import progress_bar

from cloud_diffusion.wandb import log_images, save_model

def noisify_last_frame(frames, noise_func):
    "Noisify the last frame of a sequence"
    past_frames = frames[:,:-1]
    last_frame  = frames[:,-1:]
    noise, t, e = noise_func(last_frame)
    return torch.cat([past_frames, noise], dim=1), t, e

def noisify_collate(noise_func): 
    def _inner(b): 
        "Collate function that noisifies the last frame"
        return noisify_last_frame(default_collate(b), noise_func)
    return _inner

class NoisifyDataloader(DataLoader):
    """Noisify the last frame of a dataloader by applying 
    a noise function, after collating the batch"""
    def __init__(self, dataset, *args, noise_func=None, **kwargs):
        super().__init__(dataset, *args, collate_fn=noisify_collate(noise_func), **kwargs)

class MiniTrainer:
    "A mini trainer for the diffusion process"
    def __init__(self, 
                 train_dataloader, 
                 valid_dataloader, 
                 model, 
                 sampler, 
                 device="cuda", 
                 loss_func=nn.MSELoss(), 
                 ):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_func = loss_func
        self.model = model.to(device)
        self.scaler = torch.cuda.amp.GradScaler()
        self.device = device
        self.sampler = sampler
        self.val_batch = next(iter(valid_dataloader))[0].to(device)  # grab a fixed batch to log predictions
    
    def train_step(self, loss):
        "Train for one step"
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

    def one_epoch(self, epoch=None):
        "Train for one epoch, log metrics and save model"
        self.model.train()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for batch in pbar:
            frames, t, noise = to_device(batch, device=self.device)
            with torch.autocast("cuda"):
                predicted_noise = self.model(frames, t)
                loss = self.loss_func(noise, predicted_noise)
            self.train_step(loss)
            wandb.log({"train_mse": loss.item(),
                       "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"epoch={epoch}, MSE={loss.item():2.3f}"
   
    def prepare(self, config):
        wandb.config.update(config)
        config.total_train_steps = config.epochs * len(self.train_dataloader)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, eps=1e-5)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=config.lr, total_steps=config.total_train_steps)

    def fit(self, config):
        self.prepare(config)
        self.val_batch = self.val_batch[:min(config.n_preds, 8)]  # log first 8 predictions
        for epoch in progress_bar(range(config.epochs), total=config.epochs, leave=True):
            self.one_epoch(epoch)
            
            # log predictions
            if epoch % config.log_every_epoch == 0:  
                samples = self.sampler(self.model, past_frames=self.val_batch[:,:-1])
                log_images(self.val_batch, samples)

        save_model(self.model, config.model_name)


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_device(t, device="cpu"):
    if isinstance(t, (tuple, list)):
        return [_t.to(device) for _t in t]
    elif isinstance(t, torch.Tensor):
        return t.to(device)
    else:
        raise("Not a Tensor or list of Tensors")


def ls(path: Path): 
    "Return files on Path, sorted"
    return sorted(list(path.iterdir()))


def parse_args(config):
    "A brute force way to parse arguments, it is probably not a good idea to use it"
    parser = argparse.ArgumentParser(description='Run training baseline')
    for k,v in config.__dict__.items():
        parser.add_argument('--'+k, type=type(v), default=v)
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)