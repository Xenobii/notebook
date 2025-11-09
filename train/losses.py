import torch
import torch.nn as nn


class COFProjection(nn.Module):
    def __init__(
            self,
            n_bins,
            bins_per_semitone: int,
            radius: float = 1.0
    ):
        super().__init__()

        self.bins_per_octave = 12 * bins_per_semitone
        self.n_octaves = n_bins // self.bins_per_octave

        angles = torch.arange(self.bins_per_octave, dtype=torch.float32)
        x = radius * torch.cos(2 * angles * torch.pi / self.bins_per_octave)
        y = radius * torch.sin(2 * angles * torch.pi / self.bins_per_octave)
        coords = torch.stack((x, y), dim=1)
        # [bins_per_octave, 2]
        self.register_buffer("coords", coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, pitch_classes, _ = x.shape
        # [B*nframe, pitch_classes, n_bins]
        x = x.view(B, pitch_classes, self.n_octaves, self.bins_per_octave)
        # [B*nframe, pitch_classes, n_octaves, bins_per_octave]
        x = x.mean(dim=2)
        # [B*nframe, pitch_classes, bins_per_octave]
        x = torch.matmul(x, self.coords)
        # [B*nframe, pitch_classes, 2]
        return x
    


class PitchEquivariance(nn.Module):
    def __init__(
            self,
            n_bins,
            bins_per_semitone: int,
            loss_fn: str = "huber",
            radius: float = 1.0,
            pitch_classes: int = 12,
            shift_factor: int = 7,
    ):
        super().__init__()

        if loss_fn == "huber":
            self.loss = nn.HuberLoss()
        elif loss_fn == "l1":
            self.loss = nn.L1Loss()
        elif loss_fn == "bce":
            self.loss = nn.BCELoss()
        elif loss_fn == "mse":
            self.loss = nn.MSELoss()
        else:
            raise ValueError

        self.shift_factor  = shift_factor
        self.pitch_classes = pitch_classes

        self.projection = COFProjection(
            n_bins=n_bins,
            bins_per_semitone=bins_per_semitone,
            radius=radius
        )

    @staticmethod
    def rotate_latent(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        R = torch.stack([
                torch.stack([cos, -sin]),
                torch.stack([sin, cos])
            ], dim=0).to(z)
        return z @ R.T
        
    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            shift: float
    ) -> torch.Tensor:
        # [B*nframe, pitch_class, n_bins]
        z1 = self.projection(x1)
        z2 = self.projection(x2)
        # [B*nframe, pitch_class, 2]

        theta = (shift * 2 * torch.pi / self.pitch_classes) * self.shift_factor

        z2 = self.rotate_latent(z2, theta)

        return self.loss(z1, z2)

    

class PitchClassRegularization(nn.Module):
    def __init__(
            self,
            n_bins,
            bins_per_semitone: int,
            loss_fn: str = "huber",
            radius: float = 1.0,
            pitch_classes: int = 12,
            shift_factor: int = 7
    ):
        super().__init__()

        if loss_fn == "huber":
            self.loss = nn.HuberLoss()
        elif loss_fn == "l1":
            self.loss = nn.L1Loss()
        elif loss_fn == "bce":
            self.loss = nn.BCELoss()
        elif loss_fn == "mse":
            self.loss = nn.MSELoss()
        else:
            raise ValueError

        self.shift_factor  = shift_factor
        self.pitch_classes = pitch_classes

        self.projection = COFProjection(
            n_bins=n_bins,
            bins_per_semitone=bins_per_semitone,
            radius=radius
        )

    @staticmethod
    def rotate_latent(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        R = torch.stack([
                torch.stack([cos, -sin], dim=-1),
                torch.stack([sin, cos], dim=-1)
            ], dim=-2).to(z)
        # [pitch_classes, 2, 2]
        z = torch.einsum('bpc, pcd->bpd', z, R)
        # [B*nframe, pitch_classes, 2]
        return z

    def forward(self, x):
        # [B*nframe, pitch_classes, n_bins]
        z = self.projection(x)
        # [B*nframe, pitch_classes, 2]

        theta     = (2 * torch.pi / self.pitch_classes) * self.shift_factor
        rotations = torch.arange(self.pitch_classes - 1)
        thetas    = theta * rotations
        # [pitch_classes-1]

        z_prev = self.rotate_latent(z[:, :-1, :], thetas)
        z_curr = z[:, 1:, :]

        loss = self.loss(z_curr, z_prev)

        return loss
    

class RecunstructionLoss(nn.Module):
    def __init__(
            self,
            loss_fn: str = "huber",
    ):
        super().__init__()

        if loss_fn == "huber":
            self.loss = nn.HuberLoss()
        elif loss_fn == "l1":
            self.loss = nn.L1Loss()
        elif loss_fn == "bce":
            self.loss = nn.BCELoss()
        elif loss_fn == "mse":
            self.loss = nn.MSELoss()
        else:
            raise ValueError

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor
    ):
        # [B*nframe, nbin]
        loss = self.loss(x1, x2)
        return loss 