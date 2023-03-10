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

from torcheval.metrics import Mean


from cloud_diffusion.dataset import download_dataset, CloudDataset
from cloud_diffusion.utils import init_ddpm, to_device, ddim_sampler, log_images, set_seed


PROJECT_NAME = "ddpm_clouds_debug"
DATASET_ARTIFACT = 'capecape/gtc/np_dataset:v0'

config = SimpleNamespace(    
    epochs = 1,
    model_name="unet2d",
    noise_steps=1000,
    sampler_steps=333,
    seed = 42,
    batch_size = 64,
    img_size = 64,
    device = "cuda",
    use_wandb = True,
    num_workers=8,
    num_frames=4,
    compile=True,
    lr = 5e-4,
    validation_days=3,
    n_preds=8,
    log_every_epoch = 10,
    model_params=dict(
        block_out_channels=(32, 64, 128, 256),
        norm_num_groups=8,
        in_channels=4,
        out_channels=1,
        ),
    )


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
train_dataloader = dl_ddpm(train_ds, shuffle=True)
valid_dataloader = dl_ddpm(valid_ds)

# model setup
model = UNet2DModel(**config.model_params).to(device)
init_ddpm(model)

## optim params
total_steps = config.epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-5)
scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=total_steps)
scaler = torch.cuda.amp.GradScaler()

# sampler
sampler = ddim_sampler(steps=config.sampler_steps)

# Metrics
loss_func = torch.nn.MSELoss()

def train_step(loss):
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

def one_epoch():
    model.train()
    pbar = progress_bar(train_dataloader, leave=False)
    for batch in pbar:
        with torch.autocast("cuda"):
            frames, t, noise = to_device(batch, device=device)
            predicted_noise = model(frames, t,).sample ## this if for the Diffusers's UNet2DOutput class
            loss = loss_func(noise, predicted_noise)
            train_step(loss)
            wandb.log({"train_mse": loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0]})
        pbar.comment = f"MSE={loss.item():2.3f}"


def save_model(model_name):
    "Save the model to wandb"
    model_name = f"{wandb.run.id}_{model_name}"
    models_folder = Path("models")
    if not models_folder.exists():
        models_folder.mkdir()
    torch.save(model.state_dict(), models_folder/f"{model_name}.pth")
    at = wandb.Artifact(model_name, type="model")
    at.add_file(f"models/{model_name}.pth")
    wandb.log_artifact(at)

val_batch, _, _ = next(iter(valid_dataloader))  # grab a fixed batch to log predictions
val_batch = val_batch[:config.n_preds].to(device)

def fit(config):
    for epoch in progress_bar(range(config.epochs), total=config.epochs, leave=True):
        # one_epoch(train=True)
        
        # log predicitons
        if epoch % config.log_every_epoch == 0:  
            print(val_batch.shape)
            samples = sampler(model, past_frames=val_batch[:,:-1])
            log_images(val_batch, samples)

    # save model
    save_model(config.model_name)

run = wandb.init(project=PROJECT_NAME, config=config, tags=["test_refactor"])

fit(config)

run.finish()