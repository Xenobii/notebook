import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from config.config_type import *

from typing import Optional, Tuple
import inspect

import os
import time
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchinfo import summary

from model.model import Model
from dataset.dataloader import MaestroDataset
from model.transforms import Transforms
from model.pitch_shift import PitchShift
from model.preprocessor import Preprocessor
from train.losses import PitchEquivariance, PitchClassRegularization, RecunstructionLoss


cs = ConfigStore.instance()
cs.store(name="config_type", node=NotebookConfig)


class Notebook():
    def __init__(self, cfg: NotebookConfig):
        # ======================================================================
        # 1.Torch settings
        # ======================================================================
        self.device = \
            self.init_torch(cfg.torch)
        # ======================================================================
        # 2.Dataset settings
        # ======================================================================
        self.dataloader_train, self.dataloader_valid = \
            self.init_dataloaders(cfg.dataloader)
        # ======================================================================
        # 3.Pipeline settings
        # ======================================================================
        self.preprocessor = \
            self.init_preprocessor(cfg.preprocessor)
        self.init_pitch_shifting = \
            self.init_pitch_shifting(cfg.pitch_shift)
        self.transforms = \
            self.init_transforms(cfg.transforms)
        self.model = \
            self.init_model(cfg.model)
        # ======================================================================
        # 4.Training settings
        # ======================================================================
        # self.optimizer, self.scheduler = \
        #     self.init_training(cfg.training)


    def init_torch(self, cfg: TorchConfig)-> torch.device:
        # Torch settings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == torch.device('cpu'): 
            print("WARNING: Training on cpu")
        
        torch.backends.cudnn.deterministic    = cfg.deterministic
        torch.backends.cudnn.benchmark        = cfg.benchmark
        torch.backends.cudnn.allow_tf32       = cfg.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32_mul
        return device


    def init_dataloaders(self, cfg: DataloaderConfig)-> Tuple[DataLoader, DataLoader]:
        dataset_train = MaestroDataset(cfg.dir_maestro_out, split="train")
        dataset_valid = MaestroDataset(cfg.dir_maestro_out, split="validation")
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=cfg.train.batch_size,
            shuffle=cfg.train.shuffle,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=True if cfg.train.num_workers > 0 else False,
            prefetch_factor=cfg.train.num_workers // 2 if cfg.train.num_workers > 0 else False
        )
        dataloader_valid = DataLoader(
            dataset_valid,
            batch_size=cfg.valid.batch_size,
            shuffle=cfg.valid.shuffle,
            num_workers=cfg.valid.num_workers,
            pin_memory=cfg.valid.pin_memory,
            persistent_workers=True if cfg.valid.num_workers > 0 else False,
            prefetch_factor=cfg.valid.num_workers // 2 if cfg.valid.num_workers > 0 else False
        )
        return dataloader_train, dataloader_valid

    def init_preprocessor(self, cfg: PreprocessorConfig)-> Preprocessor:
        return Preprocessor(
            cfg.sr,
            cfg.hop_length,
            cfg.fmin,
            cfg.bin_per_semitone,
            cfg.n_bins,
            cfg.center_bins,
        )
    
    def init_pitch_shifting(self, cfg: PitchShiftingConfig)-> PitchShift:
        return PitchShift(
            cfg.min_shift,
            cfg.max_shift,
            cfg.bin_per_semitone
        )
    
    def init_transforms(self, cfg: TransformsConfig)-> Transforms:
        return Transforms(
            cfg.transform_list,
            cfg.noise.min_snr,
            cfg.noise.max_snr,
            cfg.gain.min_gain,
            cfg.gain.max_gain
        )
    
    def init_model(self, cfg: ModelConfig)-> Model:
        return Model()

    def init_training(
            self,
            cfg: TrainerConfig,
            model: torch.nn.Module,
            dataloader_train: DataLoader
        )-> Tuple[Optimizer, _LRScheduler]:
        OPTIMIZERS = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "radam": optim.RAdam,
        }
        optimizer_cls = OPTIMIZERS.get(cfg.optimizer.lower())
        optimizer_kwargs = {
            "lr": cfg.lr,
            "betas": tuple(cfg.betas),
            "weight_decay": cfg.weight_decay
        }
        if "fused" in inspect.signature(optimizer_cls.__init__).parameters:
            optimizer_kwargs["fused"] = True
        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        
        # Schedulers are more dissimilar so no kwargs :(
        steps_per_epoch = len(dataloader_train) // 2
        if cfg.scheduler == "onecyclelr":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                epochs=cfg.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif cfg.scheduler == "cosineannealinglr":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.epochs,
                eta_min=cfg.lr * 1e-2,
            )
        elif cfg.scheduler == "cosineannealingwarmrestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cfg.epochs//2,
                T_mult=2,
                eta_min=cfg.lr * 1e-2,
            )
        elif cfg.scheduler == "mixed":
            warmup = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=5,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.epochs - 5,
                eta_min=cfg.lr * 1e-2,
                steps_per_epoch=steps_per_epoch
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[5]
            )
        else:
            raise ValueError
        
        return optimizer, scheduler



@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: NotebookConfig):
    trainer = Notebook(cfg)


if __name__ == "__main__":
    main()