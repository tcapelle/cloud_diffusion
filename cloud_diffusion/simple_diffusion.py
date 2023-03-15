from functools import partial

import torch
from torch import sqrt
from torch.special import expm1

from fastprogress import progress_bar

from einops import repeat

try:
    from denoising_diffusion_pytorch.simple_diffusion import right_pad_dims_to, logsnr_schedule_cosine
except:
    raise ImportError("Please install denoising_diffusion_pytorch with `pip install denoising_diffusion_pytorch`")

def q_sample(x_start, times, noise):
    log_snr = logsnr_schedule_cosine(times)

    log_snr_padded = right_pad_dims_to(x_start, log_snr)
    alpha, sigma = torch.sqrt(log_snr_padded.sigmoid()), torch.sqrt((-log_snr_padded).sigmoid())
    x_noised =  x_start * alpha + noise * sigma

    return x_noised, log_snr

def noisify_uvit(x0, pred_objective="v"):
    device = x0.device
    
    noise =  torch.randn_like(x0)
    times = torch.zeros((x0.shape[0],), device = device).float().uniform_(0, 1)
    x, log_snr = q_sample(x0, times, noise)
    
    if pred_objective == 'v':
        padded_log_snr = right_pad_dims_to(x, log_snr)
        alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
        target = alpha * noise - sigma * x0

    elif pred_objective == 'eps':
        target = noise
        
    return x, log_snr, target


    
# Sampling functions

@torch.no_grad()
def forward(model, past_frames, x, t):
    return model(torch.cat([past_frames, x], dim=1), t)


def p_mean_variance(model, past_frames, x, time, time_next, pred_objective="v"):
    
    log_snr = logsnr_schedule_cosine(time)
    log_snr_next = logsnr_schedule_cosine(time_next)
    c = -expm1(log_snr - log_snr_next)

    squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

    alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

    batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
    
    # forward pass (this is expensive)
    pred = forward(model, past_frames, x, t=batch_log_snr)

    if pred_objective == 'v':
        x_start = alpha * x - sigma * pred
    elif pred_objective == 'eps':
        x_start = (x - sigma * pred) / alpha

    x_start.clamp_(-1., 1.)

    model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

    posterior_variance = squared_sigma_next * c

    return model_mean, posterior_variance


def p_sample(model, past_frames, x, time, time_next):
    batch, *_, device = *x.shape, x.device

    model_mean, model_variance = p_mean_variance(model, past_frames, x = x, time = time, time_next = time_next)

    if time_next == 0:
        return model_mean

    noise = torch.randn_like(x)
    return model_mean + sqrt(model_variance) * noise


def p_sample_loop(model, past_frames, steps=500):
    device = past_frames.device
    new_frame = torch.randn_like(past_frames[:,-1:], dtype=past_frames.dtype, device=device)
    time_steps = torch.linspace(1., 0., steps + 1, device = device)

    for i in progress_bar(range(steps), total = steps):
        times = time_steps[i]
        times_next = time_steps[i + 1]
        new_frame = p_sample(model, past_frames, new_frame, time=times, time_next=times_next)

    new_frame.clamp_(-1., 1.)
    return new_frame


def simple_diffusion_sampler(steps=500):
    """Returns a function that samples from the diffusion model using
    the simple diffusion sampling scheme"""
    return partial(p_sample_loop, steps=steps)
