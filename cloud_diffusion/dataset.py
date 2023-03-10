from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from fastprogress import progress_bar

from cloud_diffusion.utils import ls

PROJECT_NAME = "ddpm_clouds"
ARTIFACT_NAME = 'capecape/gtc/np_dataset:v0'

class CloudDataset:
    
    def __init__(self, 
                 files, # list of numpy files to load (they come from the artifact)
                 num_frames=4, # how many consecutive frames to stack
                 scale=True, # if we images to interval [-0.5, 0.5]
                 size=64, # resize dim, original images are big (446, 780)
                 valid=False, # if True, transforms are deterministic
                ):
        
        tfms = [T.Resize((size, int(size*1.7)))] if size is not None else []
        tfms += [T.RandomCrop(size)] if not valid else [T.CenterCrop(size)]
        self.tfms = T.Compose(tfms)
        
        
        data = []
        for file in progress_bar(files, leave=False):
            one_day = np.load(file)
            if scale:
                one_day = 0.5 - self._scale(one_day)
        
            wds = np.lib.stride_tricks.sliding_window_view(
                one_day.squeeze(), 
                num_frames, 
                axis=0).transpose((0,3,1,2))
            data.append(wds)
        self.data = np.concatenate(data, axis=0)
            
    @staticmethod
    def _scale(arr):
        "Scales values of array in [0,1]"
        m, M = arr.min(), arr.max()
        return (arr - m) / (M - m)
    
    def __getitem__(self, idx):
        return self.tfms(torch.from_numpy(self.data[idx]))
    
    def __len__(self): return len(self.data)

    def save(self, fname="cloud_frames.npy"):
        np.save(fname, self.data)

    @classmethod
    def from_artifact(cls, at_name=ARTIFACT_NAME):

        with wandb.init(project=PROJECT_NAME, job_type="download_dataset"):
            artifact = wandb.use_artifact(at_name, type='dataset')
            artifact_dir = artifact.download()

        files = ls(Path(artifact_dir))

        return cls(files)

        
if __name__=="__main__":
    train_ds = CloudDataset.from_artifact(ARTIFACT_NAME)
    print(f"Let's grab 5 samples: {train_ds[0:5].shape}")