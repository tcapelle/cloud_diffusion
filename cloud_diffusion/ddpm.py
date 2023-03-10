import torch
from torch.utils.data.dataloader import default_collate


## DDPM params
## From fastai V2 Course DDPM notebooks
betamin,betamax,n_steps = 0.0001,0.02, 1000
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

