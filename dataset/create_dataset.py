import os
import json
import h5py
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import torchaudio.transforms as T
from argparse import ArgumentParser


class DatasetCreator:
    def __init__(self, config):
        self.sr = config["cqt"]["sr"]

        self.dir_maestro_in  = config["dataset"]["dir_maestro_in"]
        self.dir_maestro_out = config["dataset"]["dir_maestro_out"]

        # Init resampler for optimization
        self.resampler = None

    def create_dataset_maestro(self, verbose_flag=False):
        path_maestro_csv = os.path.join(self.dir_maestro_in, "maestro-v3.0.0.csv")
        
        df = pd.read_csv(path_maestro_csv)
        
        with h5py.File(self.dir_maestro_out, "w") as h5:
            print(f"Processing MAESTRO V3... \n")
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                f_wav = os.path.join(self.dir_maestro_in, row["audio_filename"])

                wave = self.prep_wav(f_wav)

                group = h5.create_group(f"{idx:07d}")
                group.create_dataset("wav", data=wave, compression="lzf")
                group.attrs["composer"] = str(row["canonical_composer"])
                group.attrs["title"]    = str(row["canonical_title"])
                group.attrs["split"]    = str(row["split"])
                
                if verbose_flag:
                    tqdm.write(f"Processed: {row['canonical_composer']} - {row['canonical_title']}")
            
            print(f"Finished processing, dataset saved at {self.dir_maestro_out}")


    @torch.inference_mode()
    def prep_wav(self, f_wav):
        # Resample and convert to mono
        wave, sr = torchaudio.load(f_wav)
        if self.resampler is None or self.resampler.orig_freq != sr:
            self.resampler = T.Resample(sr, self.sr)
        wave = torch.mean(wave, dim=0)
        wave = self.resampler(wave)
        return wave



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    datasetcreator = DatasetCreator(config)
    datasetcreator.create_dataset_maestro(args.verbose)
