from pathlib import Path

import wandb
import fastcore.all as fc

import torch
from torch import nn
from diffusers import UNet2DModel

try:
    from denoising_diffusion_pytorch.simple_diffusion import UViT
except:
    raise ImportError("Please install denoising_diffusion_pytorch with `pip install denoising_diffusion_pytorch`")


def init_unet(model):
    "From Jeremy's bag of tricks on fastai V2 2023"
    for o in model.down_blocks:
        for p in o.resnets:
            p.conv2.weight.data.zero_()
            for p in fc.L(o.downsamplers): nn.init.orthogonal_(p.conv.weight)

    for o in model.up_blocks:
        for p in o.resnets: p.conv2.weight.data.zero_()

    model.conv_out.weight.data.zero_()

class WandbModel:
    "A model that can be saved to wandb"
    @classmethod
    def from_checkpoint(cls, model_params, checkpoint_file):
        "Load a UNet2D model from a checkpoint file"
        model = cls(**model_params)
        print(f"Loading model from: {checkpoint_file}")
        model.load_state_dict(torch.load(checkpoint_file))
        return model

    @classmethod
    def from_artifact(cls, model_params, artifact_name):
        "Load a UNet2D model from a wandb.Artifact, need to be run in a wandb run"
        artifact = wandb.use_artifact(artifact_name, type='model')
        artifact_dir = Path(artifact.download())
        chpt_file = list(artifact_dir.glob("*.pth"))[0]
        return cls.from_checkpoint(model_params, chpt_file)

def get_unet_params(model_name="unet_small", num_frames=4):
    "Return the parameters for the diffusers UNet2d model"
    if model_name == "unet_small":
        return dict(
            block_out_channels=(16, 32, 64, 128), # number of channels for each block
            norm_num_groups=8, # number of groups for the normalization layer
            in_channels=num_frames, # number of input channels
            out_channels=1, # number of output channels
            )
    elif model_name == "unet_big":
        return dict(
            block_out_channels=(32, 64, 128, 256), # number of channels for each block
            norm_num_groups=8, # number of groups for the normalization layer
            in_channels=num_frames, # number of input channels
            out_channels=1, # number of output channels
            )
    else:
        raise(f"Model name not found: {model_name}, choose between 'unet_small' or 'unet_big'")

class UNet2D(UNet2DModel, WandbModel):
    def __init__(self, *x, **kwargs):
        super().__init__(*x, **kwargs)
        init_unet(self)

    def forward(self, *x, **kwargs):
        return super().forward(*x, **kwargs).sample ## Diffusers's UNet2DOutput class


## Simple Diffusion paper

def get_uvit_params(model_name="uvit_small", num_frames=4):
    "Return the parameters for the diffusers UViT model"
    if model_name == "uvit_small":
        return dict(
            dim=512,
            ff_mult=2,
            vit_depth=4,
            channels=4, 
            patch_size=4,
            final_img_itransform=nn.Conv2d(num_frames,1,1)
            )
    elif model_name == "uvit_big":
        return dict(
            dim=1024,
            ff_mult=4,
            vit_depth=8,
            channels=4, 
            patch_size=4,
            final_img_itransform=nn.Conv2d(num_frames,1,1)
            )
    else:
        raise(f"Model name not found: {model_name}, choose between 'uvit_small' or 'uvit_big'")

class UViTModel(UViT, WandbModel): pass