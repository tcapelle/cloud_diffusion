import random, logging
from types import SimpleNamespace

import torch, wandb
from fastprogress import progress_bar

from cloud_diffusion.dataset import download_dataset, CloudDataset
from cloud_diffusion.ddpm import UNet2D, get_unet_params, ddim_sampler
from cloud_diffusion.utils import parse_args, set_seed
from cloud_diffusion.wandb import to_video, vhtile

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    # device="cuda:0",
    device="mps",
    sampler="ddim",
    future_frames=10,  # number of future frames
    bs=8, # how many samples
)

class Inference:

    def __init__(self, config):
        self.config = config

        # create a batch of data to use for inference
        self.prepare_data()
        
        # we default to ddim as it's faster and as good as ddpm
        self.sampler = ddim_sampler(config.sampler_steps)

        # create the Unet
        model_params = get_unet_params(config.model_name, config.num_frames)

        logger.info(f"Loading model {config.model_name} from artifact: {MODEL_ARTIFACT}")
        self.model = UNet2D.from_artifact(model_params, MODEL_ARTIFACT).to(config.device)

        self.model.eval()
    
    def prepare_data(self):
        "Generates a batch of data from the validation dataset"
        logger.info(f"Downloading dataset from artifact: {DATASET_ARTIFACT}")
        files = download_dataset(DATASET_ARTIFACT, PROJECT_NAME)

        self.valid_ds = CloudDataset(files=files[-3:], # 3 days of validation data 
                                num_frames=config.num_frames, img_size=config.img_size)
        self.idxs = random.choices(range(len(self.valid_ds) - config.future_frames), k=config.bs)  # select some samples
        # fix the batch to the same samples for reproducibility
        self.batch = self.valid_ds[self.idxs].to(config.device)

    def sample_more(self, frames, future_frames=1):
        "Autoregressive sampling, starting from `frames`. It is hardcoded to work with 3 frame inputs."
        for _ in progress_bar(range(future_frames), total=future_frames, leave=True):
            # compute new frame with previous 3 frames
            new_frame = self.sampler(self.model, frames[:,-3:,...])
            # add new frame to the sequence
            frames = torch.cat([frames, new_frame.to(frames.device)], dim=1)
        return frames.cpu()

    def forecast(self):
        "Perform inference on the batch of data."
        logger.info(f"Forecasting {self.batch.shape[0]} samples for {self.config.future_frames} future frames.")
        sequences = []
        for i in range(self.config.num_random_experiments):
            logger.info(f"Generating {i+1}/{self.config.num_random_experiments} futures.")
            frames = self.sample_more(self.batch, self.config.future_frames)
            sequences.append(frames)

        return sequences

    def log_to_wandb(self, sequences):
        "Create a table with the ground truth and the generated frames. Log it to wandb."
        table = wandb.Table(columns=["id", "gt", *[f"gen_{i}" for i in range(config.num_random_experiments)], "gt/gen"])
        for i, idx in enumerate(self.idxs):
            gt_vid = to_video(self.valid_ds[idx:idx+4+config.future_frames,0,...])
            pred_vids = [to_video(frames[i]) for frames in sequences]
            gt_gen = wandb.Image(vhtile(self.valid_ds[idx:idx+4+config.future_frames,0,...], *[frames[i] for frames in sequences]))
            table.add_data(idx, gt_vid, *pred_vids, gt_gen)
        logger.info("Logging results to wandb...")
        wandb.log({f"gen_table_{config.future_frames}_random":table})

if __name__=="__main__":
    parse_args(config)
    set_seed(config.seed)

    with wandb.init(project=PROJECT_NAME, job_type=JOB_TYPE, 
                    config=config, tags=["ddpm", config.model_name]):
        infer = Inference(config)
        sequences = infer.forecast()
        infer.log_to_wandb(sequences)