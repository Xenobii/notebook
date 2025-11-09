from typing import Optional

import torch
import torch.nn as nn



class RandomNoise(nn.Module):
    def __init__(
            self,
            min_snr: float = 0.0001,
            max_snr: float = 0.01,
            p: Optional[float] = None
    ):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        snr = torch.empty(B)
        snr.uniform_(self.min_snr, self.max_snr)
        mask = torch.rand_like(snr).le(self.p)
        snr[mask] = 0

        noise_std = snr * x.view(B, -1).std(dim=-1)
        noise_std = noise_std.unsqueeze(-1).expand_as(x.view(B, -1)).view_as(x)

        noise = noise_std * torch.randn_like(x)

        return x + noise
    

class RandoimGain(nn.Module):
    def __init__(
            self,
            min_gain: float = 0.5,
            max_gain: float = 1.5,
            p: Optional[float] = None
    ):
        super().__init__()

        self.min_gain = min_gain
        self.max_gain = max_gain

        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size[0]

        vol = torch.empty(B)
        vol.uniform_(self.min_gain, self.max_gain)
        mask = torch.rand_like(vol).le(self.p)
        vol[mask] = 1
        vol = vol.unsqueeze(-1).expand_as(x.view(B, -1)).view_as(x)
        
        return vol * x
