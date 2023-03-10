from pathlib import Path
from types import SimpleNamespace

import wandb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from fastprogress import progress_bar

from diffusers import UNet2DModel

from cloud_diffusion.dataset import download_dataset, CloudDataset
from cloud_diffusion.utils import (
    save_model, get_unet_params, init_ddpm, to_device, 
    ddim_sampler, log_images, set_seed, parse_args)


PROJECT_NAME = "ddpm_clouds"
DATASET_ARTIFACT = 'capecape/gtc/np_dataset:v0'

config = SimpleNamespace(    
    epochs = 100, # number of epochs
    model_name="unet_small", # model name to save
    noise_steps=1000, # number of noise steps on the diffusion process
    sampler_steps=333, # number of sampler steps on the diffusion process
    seed = 42, # random seed
    batch_size = 6, # batch size
    img_size = 128, # image size
    device = "cuda", # device
    num_workers=8, # number of workers for dataloader
    num_frames=4, # number of frames to use as input
    lr = 5e-4, # learning rate
    validation_days=3, # number of days to use for validation
    n_preds=8, # number of predictions to make 
    log_every_epoch = 5, # log every n epochs to wandb
    )

config.model_params = get_unet_params(config.model_name, config.num_frames)


set_seed(config.seed)
device = torch.device(config.device)

## DDPM params
## From fastai V2 Course DDPM notebooks
betamin,betamax,n_steps = 0.0001,0.02,config.noise_steps
beta = torch.linspace(betamin, betamax, n_steps)
alpha = 1.-beta
alphabar = alpha.cumprod(dim=0)
sigma = beta.sqrt()

def noisify(x0, ᾱ):
    "Noise only the last frame"
    past_frames = x0[:,:-1]
    x0 = x0[:,-1:]
    device = x0.device
    n = len(x0)
    t = torch.randint(0, n_steps, (n,), dtype=torch.long)
    ε = torch.randn(x0.shape, device=device)
    ᾱ_t = ᾱ[t].reshape(-1, 1, 1, 1).to(device)
    xt = ᾱ_t.sqrt()*x0 + (1-ᾱ_t).sqrt()*ε
    return torch.cat([past_frames, xt], dim=1), t.to(device), ε

def collate_ddpm(b): 
    "Collate function that noisifies the last frame"
    return noisify(default_collate(b), alphabar)

def dl_ddpm(dataset, shuffle=True): 
    "Create a PyTorch DataLoader that noisifies the last frame"
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      collate_fn=collate_ddpm, 
                      shuffle=shuffle, 
                      num_workers=config.num_workers)



# downlaod the dataset from the wandb.Artifact
files = download_dataset(DATASET_ARTIFACT, PROJECT_NAME)
train_ds = CloudDataset(files=files[:-config.validation_days],  
                        num_frames=config.num_frames, img_size=config.img_size)
valid_ds = CloudDataset(files=files[-config.validation_days:], 
                        num_frames=config.num_frames, img_size=config.img_size)

# DDPM dataloaders
train_dataloader = dl_ddpm(train_ds, shuffle=True)
valid_dataloader = dl_ddpm(valid_ds)

# model setup
model = UNet2DModel(**config.model_params).to(device)
init_ddpm(model)

## optim params
config.total_train_steps = config.epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-5)
scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_train_steps)
scaler = torch.cuda.amp.GradScaler()

# sampler
sampler = ddim_sampler(steps=config.sampler_steps)

# Metrics
loss_func = torch.nn.MSELoss()

def train_step(loss):
    "Train for one step"
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

def one_epoch(epoch=None):
    "Train for one epoch, log metrics and save model"
    model.train()
    pbar = progress_bar(train_dataloader, leave=False)
    for batch in pbar:
        frames, t, noise = to_device(batch, device=device)
        with torch.autocast("cuda"):
            predicted_noise = model(frames, t,).sample ## Diffusers's UNet2DOutput class
            loss = loss_func(noise, predicted_noise)
        train_step(loss)
        wandb.log({"train_mse": loss.item(),
                   "learning_rate": scheduler.get_last_lr()[0]})
        pbar.comment = f"epoch={epoch}, MSE={loss.item():2.3f}"

val_batch, _, _ = next(iter(valid_dataloader))  # grab a fixed batch to log predictions
val_batch = val_batch[:min(config.n_preds, 8)].to(device)

def fit(config):
    for epoch in progress_bar(range(config.epochs), total=config.epochs, leave=True):
        one_epoch(epoch)
        
        # log predicitons
        if epoch % config.log_every_epoch == 0:  
            samples = sampler(model, past_frames=val_batch[:,:-1])
            log_images(val_batch, samples)

    save_model(model, config.model_name)

if __name__=="__main__":
    parse_args(config)
    run = wandb.init(project=PROJECT_NAME, config=config, tags=["test_refactor", config.model_name])
    fit(config)
    run.finish()