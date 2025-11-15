from dataclasses import dataclass, field
from typing import List, Dict, Any

# ==============================================================================
# Dataset
# ==============================================================================
@dataclass
class DatasetConfig:
    dir_maestro_in: str
    dir_maestro_out: str
# ==============================================================================
# Torch
# ==============================================================================
@dataclass
class TorchConfig:
    gpu: bool
    deterministic: bool
    benchmark: bool
    allow_tf32: bool
    allow_tf32_mul: bool
# ==============================================================================
# Dataloader
# ==============================================================================
@dataclass
class DataloaderSplitConfig:
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
@dataclass
class DataloaderConfig:
    dir_maestro_in: str
    dir_maestro_out: str
    train: DataloaderSplitConfig
    valid: DataloaderSplitConfig
# ==============================================================================
# Preprocessing
# ==============================================================================
@dataclass
class PreprocessorConfig:
    sr: int
    hop_length: int
    fmin: float 
    bin_per_semitone: int
    n_bins: int
    center_bins: bool
    gamma: int
    center: bool
# ==============================================================================
# Pitch shift
# ==============================================================================
@dataclass
class PitchShiftingConfig:
    min_shift: int
    max_shift: int
    bin_per_semitone: int
# ==============================================================================
# Transforms
# ==============================================================================
@dataclass
class GainConfig:
    min_gain: float
    max_gain: float
@dataclass
class NoiseConfig:
    min_snr: float
    max_snr: float
@dataclass
class TransformsConfig:
    transform_list: List[str]
    gain: GainConfig
    noise: NoiseConfig
# ==============================================================================
# Model
# ==============================================================================
@dataclass
class FramePaddingConfig:
    pad_len: int
    pad_value: float
@dataclass
class OctavePoolingConfig:
    bins_per_octave: int
@dataclass
class EncoderConfig:
    len_margin: int
    pitch_classes: int
@dataclass
class PredictorConfig:
    pitch_classes: int
    n_bins: int
    len_margin: int
@dataclass
class ModelConfig:
    frame_padding: FramePaddingConfig
    octave_pooling: OctavePoolingConfig
    encoder: EncoderConfig
    predictor: PredictorConfig
# ==============================================================================
# Training
# ==============================================================================
@dataclass
class TrainerConfig:
    optimizer: str
    scheduler: str
    lr: float
    betas: List[float]
    weight_decay: float
    batch_size: int
    epochs: int
    num_workers: int
# ==============================================================================
# Criterions
# ==============================================================================
@dataclass
class PitchEquivarianceConfig:
    n_bins: int

@dataclass
class ReconstructionLossConfig:
    loss_fn: str

@dataclass
class CriterionConfig:
    pitch_equivariance: PitchEquivarianceConfig
    pitch_class_reqularization: PitchEquivarianceConfig
    reconstruction_loss: ReconstructionLossConfig
# ==============================================================================
# Notebook
# ==============================================================================
@dataclass
class NotebookConfig:
    dataset: DatasetConfig
    torch: TorchConfig
    dataloader: DataloaderConfig
    preprocessor: PreprocessorConfig
    pitch_shift: PitchShiftingConfig
    transforms: TransformsConfig
    model: ModelConfig
    training: TrainerConfig
    criterions: CriterionConfig