import os
import torch
import librosa
from torch.utils.data import Dataset

from model import STFTModule

class AudioDataset(Dataset):
    """AudioDataset"""

    def __init__(self, dataset_path, sr):
        """
        Initializes the dataset with the path to the dataset and the sampling rate.
        """
        self.sr = sr
        self.mix_dir = os.path.join(dataset_path, "mix")
        self.target_dir = os.path.join(dataset_path, "clean")
        self.mix = sorted(os.listdir(self.mix_dir))
        self.target = sorted(os.listdir(self.target_dir))
        self.stft = STFTModule()
        
    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        mix_path = os.path.join(self.mix_dir, self.mix[idx])
        target_path = os.path.join(self.target_dir, self.target[idx])
        
        mix_np, _ = librosa.load(mix_path, sr=self.sr)
        target_np, _ = librosa.load(target_path, sr=self.sr)
        
        mag_mix , _ = self.stft(torch.tensor(mix_np))
        mag_target , _ = self.stft(torch.tensor(target_np))
        
        mag_noise = mag_mix - mag_target
        
        return torch.tensor(mix_np), mag_noise
