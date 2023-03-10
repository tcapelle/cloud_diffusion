import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


from denoising_diffusion_pytorch.simple_diffusion import UViT, right_pad_dims_to, logsnr_schedule_cosine

def q_sample(x_start, times, noise):
    log_snr = logsnr_schedule_cosine(times)

    log_snr_padded = right_pad_dims_to(x_start, log_snr)
    alpha, sigma = torch.sqrt(log_snr_padded.sigmoid()), torch.sqrt((-log_snr_padded).sigmoid())
    x_noised =  x_start * alpha + noise * sigma

    return x_noised, log_snr

def noisify(frames, pred_objective="v"):
    past_frames = frames[:,:-1]
    last_frame  = frames[:,-1:]
    device = frames.device
    
    noise =  torch.randn_like(last_frame)
    times = torch.zeros((last_frame.shape[0],), device = device).float().uniform_(0, 1)
    x, log_snr = q_sample(last_frame, times, noise)
    
    if pred_objective == 'v':
        padded_log_snr = right_pad_dims_to(x, log_snr)
        alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
        target = alpha * noise - sigma * last_frame

    elif pred_objective == 'eps':
        target = noise
        
    return torch.cat([past_frames, x], dim=1), log_snr, target

def collate_simple_diffusion(b): 
    "Collate function that noisifies the last frame"
    return noisify(default_collate(b))

def dl_ddpm(dataset, shuffle=True): 
    "Create a PyTorch DataLoader that noisifies the last frame"
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      collate_fn=collate_ddpm, 
                      shuffle=shuffle, 
                      num_workers=config.num_workers)

uvit = UViT(512, 
            ff_mult=2,
            vit_depth=4,
            channels=4, 
            patch_size=4,
            final_img_itransform=nn.Conv2d(4,1,1))