from typing import Optional, List, Tuple

import torch
import torch.nn as nn


# Based on the PESTO implementation
class PitchShift(nn.Module):
    def __init__(
            self,
            min_shift: int,
            max_shift: int,
            bins_per_semitone: int
    ):
        super().__init__()
        self.min_shift = min_shift * bins_per_semitone
        self.max_shift = max_shift * bins_per_semitone
        
        self.lower_bin = self.max_shift 

    def forward(self, spec: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, int]:
        # [B, nframe, nbin]
        B, _, n_bin = spec.shape

        n_bin_out = n_bin - self.max_shift + self.min_shift
        n_shifts = torch.randint(self.min_shift, self.max_shift+1, (B,), device=spec.device)
        
        x = spec[..., self.lower_bin: self.lower_bin + n_bin_out]
        x_shifted = self.extract_bins(spec, self.lower_bin - n_shifts, n_bin_out)

        return x, x_shifted, n_shifts
    
    @staticmethod
    def extract_bins(x: torch.Tensor, first_bin: torch.LongTensor, n_bin_out: int) -> torch.Tensor:
        B = x.size(0)
        bins = first_bin.unsqueeze(-1) + torch.arange(n_bin_out, device=x.device)
        bins = bins.view(B, 1, n_bin_out).expand(B, x.size(1), n_bin_out)
        return x.gather(-1, bins)
