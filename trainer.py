import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from typing import Tuple, Dict
import logging

from model import UNet
from dataset import AudioDataset
from losses import SpeechEnhancementLoss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int, device: torch.device = torch.device('cuda'), model_save_path: str = None, model_load_path: str = None) -> None:
        """
        Initialize the Trainer class.

        Args:
            model (nn.Module): The model to be trained.
            criterion (nn.Module): The loss function.
            optimizer (optim.Optimizer): The optimizer.
            num_epochs (int): The number of training epochs.
            device (torch.device, optional): The device to be used for training. Defaults to 'cuda'.
            model_save_path (str, optional): The path to save the trained model. Defaults to None.
            model_load_path (str, optional): The path to load a pre-trained model. Defaults to None.
        """
        self.device = device
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path
        self.scaler = torch.cuda.amp.GradScaler()

        self.best_val_loss = float('inf')
        
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        logger.info(f'Model save directory: {model_save_path}')
        
    def _train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> float:
        """
        Perform one training epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader for training data.
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for mix, target in tqdm(dataloader, desc=f'Training Epoch {epoch+1}/{self.num_epochs}', leave=False, ncols=150, position=0):
            mix, target = mix.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs, _ = self.model(mix)
                target = target.unsqueeze(1)
                
                loss = self.criterion(outputs.to(self.device), target)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * mix.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> float:
        """
        Perform one evaluation epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader for evaluation data.
            epoch (int): The current epoch number.

        Returns:
            float: The average validation loss for the epoch.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mix, target in tqdm(dataloader, desc=f'Validation Epoch {epoch+1}/{self.num_epochs}', leave=False, ncols=150, position=0):
                mix, target = mix.to(self.device), target.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs, _ = self.model(mix)
                    target = target.unsqueeze(1)
                    loss = self.criterion(outputs.to(self.device), target)

                val_loss += loss.item() * mix.size(0)

        val_loss /= len(dataloader.dataset)
        return val_loss

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save the model checkpoint.

        Args:
            epoch (int): The current epoch number.
            val_loss (float): The validation loss at the current epoch.
        """
        checkpoint_path = os.path.join(self.model_save_path, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        logger.info(f'Saved best model with validation loss: {val_loss:.4f} at epoch {epoch+1}')

    def _load_checkpoint(self, model_load_path: str) -> Tuple[int, float]:
        """
        Load a model checkpoint.

        Args:
            model_load_path (str): The path to the model checkpoint.

        Returns:
            Tuple[int, float]: The epoch number and validation loss from the loaded checkpoint.
        """
        checkpoint = torch.load(model_load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info(f'Loaded checkpoint from {model_load_path} at epoch {checkpoint["epoch"]+1} with validation loss {checkpoint["val_loss"]:.4f}')
        return checkpoint['epoch'], checkpoint['val_loss']

    def train_and_eval(self, dataloaders: Dict[str, torch.utils.data.DataLoader]) -> None:
        """
        Perform training and evaluation.

        Args:
            dataloaders (Dict[str, torch.utils.data.DataLoader]): A dictionary of dataloaders for training and evaluation data.
        """
        torch.backends.cudnn.benchmark = True
        start_epoch = 0
        
        if self.model_load_path:
            start_epoch, _ = self._load_checkpoint(self.model_load_path)
            start_epoch += 1
            logger.info(f'Model loaded and training continued from epoch {start_epoch}')
        
        for epoch in range(start_epoch, self.num_epochs):
            # Train
            train_loss = self._train_epoch(dataloaders["train"], epoch)
            logger.info(f'Epoch {epoch+1}/{self.num_epochs}, Training Loss: {train_loss:.4f}')

            # Evaluate
            val_loss = self._eval_epoch(dataloaders["eval"], epoch)
            logger.info(f'Epoch {epoch+1}/{self.num_epochs}, Validation Loss: {val_loss:.4f}')

            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss)

if __name__ == "__main__":
    
    device = torch.device("cuda")
    model = UNet()
    model.to(device)
    sr = 16000
    batch_size = {
        "train": 8,
        "eval": 4
    }
    learning_rate = 0.0001
    num_epochs = 10
    
    datasets = {
        "train": AudioDataset("dev_dataset/train", sr),
        "eval":  AudioDataset("dev_dataset/eval", sr)
    }
    
    dataloaders = {k: torch.utils.data.DataLoader(datasets[k], batch_size[k]) for k, v in datasets.items()}

    # criterion = SpeechEnhancementLoss()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the Trainer
    trainer = Trainer(model, criterion, optimizer, num_epochs, device, "trainedModel")
    
    logger.info('Starting training and evaluation...')
    trainer.train_and_eval(dataloaders)
    logger.info('Training and evaluation completed.')
