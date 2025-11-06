import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler
from torchinfo import summary

from model.model import Model_V1
from model.preprocessor import Preprocessor
from dataset.dataloader import MaestroDataset


class Trainer():
    def __init__(
            self,
            config
        ):
        self.f_dataset = config["dataset"]["dir_maestro_out"]

        self.harmonics  = config["cqt"]["harmonics"]
        self.n_bins     = config["cqt"]["n_bins"]
        self.len_margin = config["cqt"]["len_margin"]
        
        optimizer_name   = config["training"]["optimizer"]
        scheduler_name   = config["training"]["scheduler"]
        self.lr          = config["training"]["lr"]
        self.epochs      = config["training"]["epochs"]
        self.batch_size  = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]

        
        # Torch settings
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("Training model on cuda")
        else:
            self.device = 'cpu'
            print("WARNING: Training model on cpu!")

        torch.backends.cudnn.deterministic    = True
        torch.backends.cudnn.benchmark        = False
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cuda.matmul.allow_tf32 = True

        # Prep + Model settings
        self.prep = Preprocessor(config)
        self.prep_info = summary(
            self.prep,
            input_size=(1, 512),
            col_names=("input_size", "output_size", "num_params"),
            device=self.device
        )
        print(f"Preprocessor: \n {self.prep_info}")

        self.model = Model_V1(config)
        self.model = self.model.to(self.device)
        self.model_info = summary(
            self.model,
            input_size=(1, self.harmonics, self.n_bins, 2*self.len_margin+1),
            col_names=("input_size", "output_size", "num_params"),
            device=self.device
        )
        print(f"Model       : \n {self.prep_info}")

        # Dataloader
        dataset_train = MaestroDataset(self.f_dataset, split="train")
        dataset_valid = MaestroDataset(self.f_dataset, split="validation")

        self.dataloader_train = DataLoader(
            dataset_train,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.num_workers//2 if self.num_workers > 0 else False
        ) 
        self.dataloader_valid = DataLoader(
            dataset_valid,
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=self.num_workers//2 if self.num_workers > 0 else False
        )
        
        # Training settings
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                fused=True if self.device=='cuda' else False
            )
        if optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                fused=True if self.device=='cuda' else False
            )
        elif optimizer_name == "radam":
            self.optimizer = optim.RAdam(
                self.model.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
        else: 
            raise ValueError
        print(f"Optimizer   : {optimizer_name}")
        
        steps_per_epoch = len(self.dataloader_train) // 2
        if scheduler_name == "onecyclelr":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        elif scheduler_name == "cosineannealinglr":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.lr * 1e-2,
                steps_per_epoch=steps_per_epoch
            )
        elif scheduler_name == "cosineannealingwarmrestarts":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.epochs//2,
                T_mult=2,
                eta_min=self.lr * 1e-2,
                steps_per_epoch=steps_per_epoch
            )
        elif scheduler_name == "mixed":
            warmup = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-3,
                total_iters=5,
                steps_per_epoch=steps_per_epoch
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - 5,
                eta_min=self.lr * 1e-2,
                steps_per_epoch=steps_per_epoch
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[5]
            )
        else:
            raise ValueError
        
        print(f"Scheduler   : {scheduler_name}")


        # Criterions

        # LATER
        



