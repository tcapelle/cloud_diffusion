from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from fastprogress import progress_bar

from cloud_diffusion.utils import ls

PROJECT_NAME = "ddpm_clouds"
DATASET_ARTIFACT = 'capecape/gtc/np_dataset:v0'

class DummyNextFrameDataset:
    "Dataset that returns random images"
    def __init__(self, num_frames=4, img_size=64, N=1000):
        self.img_size = img_size
        self.num_frames = num_frames
        self.N = N

    def __getitem__(self, idx):
        return torch.randn(self.num_frames, self.img_size, self.img_size)
    
    def __len__(self):
        return self.N


class CloudDataset:
    """Dataset for cloud images
    It loads numpy files from wandb artifact and stacks them into a single array
    It also applies some transformations to the images
    """
    def __init__(self, 
                 files, # list of numpy files to load (they come from the artifact)
                 num_frames=4, # how many consecutive frames to stack
                 scale=True, # if we images to interval [-0.5, 0.5]
                 img_size=64, # resize dim, original images are big (446, 780)
                 valid=False, # if True, transforms are deterministic
                ):
        
        tfms = [T.Resize((img_size, int(img_size*1.7)))] if img_size is not None else []
        tfms += [T.RandomCrop(img_size)] if not valid else [T.CenterCrop(img_size)]
        self.tfms = T.Compose(tfms)
        self.load_data(files, num_frames, scale)
        
    def load_day(self, file, scale=True):
        one_day = np.load(file)
        if scale:
            one_day = 0.5 - self._scale(one_day)
        return one_day

    def load_data(self, files, num_frames, scale):
        "Loads all data into a single array self.data"
        data = []
        for file in (pbar:=progress_bar(files, leave=False)):
            one_day = self.load_day(file, scale)
            wds = np.lib.stride_tricks.sliding_window_view(
                one_day.squeeze(), 
                num_frames, 
                axis=0).transpose((0,3,1,2))
            data.append(wds)
            pbar.comment = f"Creating CloudDataset from {file}"
        self.data = np.concatenate(data, axis=0)

    def shuffle(self):
        """Shuffles the dataset, useful for getting 
        interesting samples on the validation dataset"""
        idxs = torch.randperm(len(self.data))
        self.data = self.data[idxs]
        return self

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


class CloudDatasetInference(CloudDataset):
     def load_data(self, files, num_frames=None, scale=None):
        "Loads all data into a single array self.data"
        data = []
        max_length = 100
        for file in files:
            one_day = self.load_day(file, scale)
            data.append(one_day)
            max_length = min(max_length, len(one_day))
        self.data = np.stack([d[:max_length] for d in data], axis=0).squeeze()


def download_dataset(at_name, project_name):
    "Downloads dataset from wandb artifact"
    def _get_dataset(run):
        artifact = run.use_artifact(at_name, type='dataset')
        return artifact.download()

    if wandb.run is not None:
        run = wandb.run
        artifact_dir = _get_dataset(run)
    else:
        run = wandb.init(project=project_name, job_type="download_dataset")
        artifact_dir = _get_dataset(run)
        run.finish()

    files = ls(Path(artifact_dir))
    return files

if __name__=="__main__":
    files = download_dataset(DATASET_ARTIFACT, project_name=PROJECT_NAME)
    train_ds = CloudDataset(files)
    print(f"Let's grab 5 samples: {train_ds[0:5].shape}")