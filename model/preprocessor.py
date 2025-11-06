import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.Spectrogram import CQT2010v2
from typing import Optional



class Preprocessor(nn.Module):
    def __init__(self, config):
        super().__init__()

        harmonics         = config["cqt"]["harmonics"]
        len_margin        = config["cqt"]["len_margin"]
        sr                = config["cqt"]["sr"]
        hop_length        = config["cqt"]["hop_length"]
        fmin              = config["cqt"]["fmin"]
        bins_per_semitone = config["cqt"]["bins_per_semitone"]
        n_bins            = config["cqt"]["n_bins"]
        center_bins       = config["cqt"]["center_bins"]
        gamma             = config["cqt"]["gamma"]
        center            = config["cqt"]["center"]

        eps = torch.finfo(torch.float32).eps

        self.hcqt = HCQT(
            harmonics=harmonics,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            bins_per_semitone=bins_per_semitone,
            n_bins=n_bins,
            center_bins=center_bins,
            gamma=gamma,
            center=center
        )

        self.tolog = ToLogMagnitude(eps)

        self.framepad = FramePadding(
            pad_len=len_margin,
            pad_value=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, nframe]
        x = self.hcqt(x)
        # [B, harmonics, nbin, nframe, 2]
        x.permute(0, 3, 1, 2, 4)
        # [B, nframe, harmonics, nbin, 2]
        x = self.tolog(x)
        # [B, nframe, harmonics, nbin]
        x = self.framepad(x)
        # [B, nframe, harmonics, nbin, 2M+1]
        x = x.flatten(0, 1)
        # [B*nframe, harmonics, nbin, 2M+1]
        return x


    
class HCQT(nn.Module):
    def __init__(
            self,
            harmonics=8,
            sr: int=16000,
            hop_length: int = 512,
            fmin: float = 32.7,
            fmax: Optional[float] = None,
            bins_per_semitone: int = 3,
            n_bins: int = 84,
            center_bins: bool = True,
            gamma: int = 0,
            center: bool = True,
            **cqt_args
    ):
        super().__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.cqt_kernels = nn.ModuleList([
            CQT2010v2(
                sr=sr,
                hop_length=hop_length,
                fmin=h*fmin,
                fmax=fmax,
                n_bins=n_bins,
                bins_per_octave=12*bins_per_semitone,
                gamma=gamma,
                center=center,
                output_format="Complex",
                **cqt_args
            )
            for h in harmonics
        ])

    def froward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, nframes]
        x = torch.stack([cqt(x) for cqt in self.cqt_kernels], dim=1)
        # [B, harmonics, nbin, nframe, 2]
        return x



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



class FramePadding(nn.Module):
    def __init__(self, pad_len, pad_value):
        super().__init__()

        self.pad_len   = pad_len
        self.pad_value = pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, nframe, harmonics, nbin]
        x = F.pad(x, (0, 0, 0, 0, self.pad_len, self.pad_len), value=self.pad_value)
        # [B, nframe+2M, harmonics, nbin]
        x = x.unfold(dimension=1, size=2*self.pad_len + 1, step=1)
        # [B, nframe, harmonics, nbin, 2M+1]
        return x