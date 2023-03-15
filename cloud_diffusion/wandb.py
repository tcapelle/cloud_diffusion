from pathlib import Path

import torch
import wandb
import numpy as np

## For Training

def to_wandb_image(img):
    "Convert a tensor to a wandb.Image"
    return wandb.Image(torch.cat(img.split(1), dim=-1).cpu().numpy())

def log_images(xt, samples):
    "Log sampled images to wandb"
    device = samples.device
    frames = torch.cat([xt[:, :-1,...].to(device), samples], dim=1)
    wandb.log({"sampled_images": [to_wandb_image(img) for img in frames]})

def save_model(model, model_name):
    "Save the model to wandb"
    model_name = f"{wandb.run.id}_{model_name}"
    models_folder = Path("models")
    if not models_folder.exists():
        models_folder.mkdir()
    torch.save(model.state_dict(), models_folder/f"{model_name}.pth")
    at = wandb.Artifact(model_name, type="model")
    at.add_file(f"models/{model_name}.pth")
    wandb.log_artifact(at)


## For Inference
def htile(img):
    "Horizontally tile a batch of images."
    return torch.cat(img.split(1), dim=-1)

def vtile(img):
    "Vertically tile a batch of images."
    return torch.cat(img.split(1), dim=-2)

def vhtile(*imgs):
    "Vertically and horizontally tile a batch of images."
    return vtile(torch.cat([htile(img) for img in imgs], dim=0))

def scale(arr):
    "Scales values of array in [0,1]"
    m, M = arr.min(), arr.max()
    return (arr - m) / (M - m)

def preprocess_frames(data):
    "Preprocess frames for wandb.Video"
    sdata = scale(data.squeeze())
    # print(sdata.shape)
    def tfm(frame):
        rframe = 255 * frame
        return rframe.cpu().numpy().astype(np.uint8)
    return [tfm(frame) for frame in sdata]

def to_video(data):
    "create wandb.Video container"
    frames = preprocess_frames(data)
    vid = np.stack(frames)[:, None, ...]
    return wandb.Video(vid)