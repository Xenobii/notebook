import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features import CQT2010v2
from typing import Optional


class Preprocessor(nn.Module):
    def __init__(
            self,
            sr: int,
            hop_length: int,
            fmin: float,
            bins_per_semitone: int,
            n_bins: int,
            center_bins: bool,
        ):
        super().__init__()
        eps = torch.finfo(torch.float32).eps

        self.cqt = CQT(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            bins_per_semitone=bins_per_semitone,
            n_bins=n_bins,
            center_bins=center_bins,
        )

        self.tolog = ToLogMagnitude(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, nframe]
        x = self.cqt(x)
        # [B, nbin, nframe, 2]
        x.permute(0, 3, 1, 2, 4)
        # [B, nframe, nbin, 2]
        x = self.tolog(x)
        # [B, nframe, nbin]
        return x
    

class CQT(nn.Module):
    def __init__(
            self,
            sr: int = 16000,
            hop_length: int = 512,
            fmin: float = 32.7,
            fmax: Optional[float] = None,
            bins_per_semitone: int = 3,
            n_bins: int = 84,
            center_bins: bool = True,
            **cqt_args
    ):
        super().__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.cqt = CQT2010v2(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_bins=n_bins,
            bins_per_octave=12*bins_per_semitone,
            output_format="Complex",
            verbose=False,
            **cqt_args
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.cqt(wav)


class ToLogMagnitude(nn.Module): 
    def __init__(self, min_val):
        super().__init__()
        self.eps = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 2:
            x = torch.sqrt(x[..., 0]**2 + x[..., 1]**2)
        else:
            x = x.abs()
        x.clamp_(min=self.eps).log10_().mul_(20)
        return x
