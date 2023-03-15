from types import SimpleNamespace

import wandb
import torch
from torch.utils.data import DataLoader

from cloud_diffusion.dataset import download_dataset, CloudDataset
from cloud_diffusion.utils import MiniTrainer, set_seed, parse_args
from cloud_diffusion.simple_diffusion import collate_simple_diffusion, simple_diffusion_sampler
from cloud_diffusion.models import UViT, get_uvit_params

DEBUG = True
PROJECT_NAME = "ddpm_clouds"
DATASET_ARTIFACT = 'capecape/gtc/np_dataset:v1'

config = SimpleNamespace(    
    epochs = 100, # number of epochs
    model_name="uvit_small", # model name to save
    strategy="simple_diffusion", # strategy to use [ddpm, simple_diffusion]
    noise_steps=1000, # number of noise steps on the diffusion process
    sampler_steps=500, # number of sampler steps on the diffusion process
    seed = 42, # random seed
    batch_size = 6, # batch size
    img_size = 512, # image size
    device = "cuda", # device
    num_workers=8, # number of workers for dataloader
    num_frames=4, # number of frames to use as input
    lr = 5e-4, # learning rate
    validation_days=3, # number of days to use for validation
    n_preds=8, # number of predictions to make 
    log_every_epoch = 5, # log every n epochs to wandb
    )

def train_func(config):
    config.model_params = get_uvit_params(config.model_name, config.num_frames)

    set_seed(config.seed)
    device = torch.device(config.device)

    # downlaod the dataset from the wandb.Artifact
    files = download_dataset(DATASET_ARTIFACT, PROJECT_NAME)

    files = files[0:5] if DEBUG else files
    train_days, valid_days = files[:-config.validation_days], files[-config.validation_days:]
    train_ds = CloudDataset(files=train_days, num_frames=config.num_frames, img_size=config.img_size)
    valid_ds = CloudDataset(files=valid_days, num_frames=config.num_frames, img_size=config.img_size).shuffle()

    collate_fn = collate_simple_diffusion

    # DDPM dataloaders
    train_dataloader = DataLoader(train_ds, config.batch_size, shuffle=True, 
                                  collate_fn=collate_fn,  num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_ds, config.batch_size, shuffle=False, 
                                  collate_fn=collate_fn,  num_workers=config.num_workers)

    # model setup
    model = UViT(**config.model_params)

    # sampler
    sampler = simple_diffusion_sampler(steps=config.sampler_steps)

    # A simple training loop
    trainer = MiniTrainer(train_dataloader, valid_dataloader, model, sampler, device)
    trainer.fit(config)

if __name__=="__main__":
    parse_args(config)
    with wandb.init(project=PROJECT_NAME, config=config, tags=["sd", config.model_name]):
        train_func(config)