from typing import Optional, List

import torch
import torch.nn as nn


class Transforms(nn.Module):
    def __init__(
            self,
            transforms: List[str],
            min_snr: Optional[float] = None,
            max_snr: Optional[float] = None,
            min_gain: Optional[float] = None,
            max_gain: Optional[float] = None
    ):
        super().__init__()

        transform_modules = {
            "noise": RandomNoise(min_snr=min_snr, max_snr=max_snr),
            "gain": RandomGain(min_gain=min_gain, max_gain=max_gain)
        }

        transform_list = []
        for name in transforms:
            if name not in transform_modules:
                raise ValueError(f"Unknown transform: {name}")
            transform_list.append(transform_modules[name])
        
        self.transforms = nn.Sequential(*transform_list)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.transforms(x)



class RandomNoise(nn.Module):
    def __init__(
            self,
            min_snr: float = 0.0001,
            max_snr: float = 0.01
    ):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        snr = torch.empty(B)
        snr.uniform_(self.min_snr, self.max_snr)

        noise_std = snr * x.view(B, -1).std(dim=-1)
        noise_std = noise_std.unsqueeze(-1).expand_as(x.view(B, -1)).view_as(x)

        noise = noise_std * torch.randn_like(x)

        return x + noise
    

class RandomGain(nn.Module):
    def __init__(
            self,
            min_gain: float = 0.5,
            max_gain: float = 1.5
    ):
        super().__init__()

        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        vol = torch.empty(B)
        vol.uniform_(self.min_gain, self.max_gain)
        vol = vol.unsqueeze(-1).expand_as(x.view(B, -1)).view_as(x)
        
        return vol * x