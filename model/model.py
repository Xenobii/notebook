import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from typing import Optional, List, Tuple


class Model(nn.Module):
    def __init__(
            self,
            padding_fn: nn.Module,
            octavepooling: nn.Module,
            predictor: nn.Module,
            encoder: nn.Module,
        ):
        super().__init__()
        self.framepad      = padding_fn
        self.octavepooling = octavepooling
        self.encoder       = encoder
        self.predictor     = predictor

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        B, nframe, nbin = x.shape
        # [B, nframe, nbin]
        x = self.framepad(x)
        # [B, nframe, nbin, 2M+1]
        x = x.flatten(0, 1)
        # [B*nframe, nbin, 2M+1]
        x_ = x

        # BRANCH 1
        x = self.encoder(x)
        # [B*nframe, channels, nbin]

        notebook = self.octavepooling(x)
        # [B*nframe, pitch_classes, bins_per_octave]
        
        # BRANCH 2
        predictor = self.predictor(x_)
        # [B*nframe, pitch_classes]

        reconstruction = (predictor @ notebook)
        # [B, nframe, nbin]
        reconstruction = reconstruction.reshape(B, nframe, nbin)
        # [B*nframe, nbin]
        return notebook, predictor, reconstruction


class DemoEncoder(nn.Module): 
    def __init__(
        self,
        len_margin: int,
        pitch_classes: int
    ):
        super().__init__()

        self.convframe = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(1, 2*len_margin+1),
            padding=(0, 0),
            stride=1
        )

        self.convbin = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.convpitchclass = nn.Conv1d(
            in_channels=64,
            out_channels=pitch_classes,
            kernel_size=1,
            padding=0,
            stride=1
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # [B, nbin, 2M+1]
        x = x.unsqueeze(1)
        # [B, 1, nbin, 2M+1]
        x = self.convframe(x)
        # [B, channels, nbin, 1]
        x = x.squeeze(3)
        # [B, channels, nbin]
        x = self.convpitchclass(x)
        # [B, pitch_classes, nbin]
        return x


class DemoPredictor(nn.Module):
    def __init__(
            self,
            pitch_classes: int,
            n_bins: int,
            len_margin: int
    ):
        super().__init__()
        self.pitch_classes = pitch_classes
        self.n_bins        = n_bins
        self.margin_dim    = 2*len_margin+1

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1)
        )

        self.pool = nn.MaxPool2d(
            kernel_size=(self.n_bins, self.margin_dim)
        )

        self.fc = nn.Linear(128, pitch_classes)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # [B, nbin, 2M+1]
        x = x.unsqueeze(1)
        # [B, 1, n_bin, 2M+1]
        x = self.conv(x)
        # [B, ch_1, n_bin, 2M+1]
        x = self.pool(x)
        # [B, ch_1, 1, 1]
        x = x.squeeze(3).squeeze(2)
        # [B, ch_1]
        x = self.fc(x)
        # [B, pitch_class]
        return x


class OctavePooling(nn.Module):
    def __init__(
            self,
            bins_per_octave: int,
    ):
        super().__init__()
        self.bins_per_octave = bins_per_octave

    @torch.inference_mode()
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # [B, channels, n_bins]
        x = rearrange("B C (j k) -> B C k j", k=self.bins_per_octave)
        # [B, channels, bins_per_octave, n_octaves]
        x = x.mean(dim=2)
        # [B, channels, bins_per_octave]
        return x


class FramePadding(nn.Module):
    def __init__(self, pad_len, pad_value):
        super().__init__()
        self.pad_len   = pad_len
        self.pad_value = pad_value

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, nframe, nbin]
        x = F.pad(x, (0, 0, 0, 0, self.pad_len, self.pad_len), value=self.pad_value)
        # [B, nframe+2M, nbin]
        x = x.unfold(dimension=1, size=2*self.pad_len + 1, step=1)
        # [B, nframe, nbin, 2M+1]
        return x