"""Base trainer class for model training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base class for all trainers in the framework."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize base trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer for training.
            criterion: Loss function.
            device: Device to train on.
            scheduler: Learning rate scheduler.
            metrics: List of metrics to compute.
            **kwargs: Additional trainer arguments.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.metrics = metrics or ["mse", "mae", "r2"]
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
        
        # Early stopping
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 30)
        self.patience_counter = 0
        
        logger.info(f"Initialized trainer for {model.__class__.__name__}")
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics.
        """
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics.
        """
        pass
    
    def train(
        self,
        num_epochs: int,
        save_best_model: bool = True,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train.
            save_best_model: Whether to save the best model.
            save_path: Path to save the model.
            **kwargs: Additional training arguments.
            
        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # Validation phase
            val_metrics = self.validate_epoch()
            self.val_history.append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Check for best model
            val_loss = val_metrics['loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if save_best_model and save_path:
                    self._save_model(save_path, epoch, val_metrics)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'epochs_trained': self.current_epoch + 1,
        }
    
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            
        Returns:
            Dictionary with computed metrics.
        """
        metrics_dict = {}
        
        for metric in self.metrics:
            if metric.lower() == 'mse':
                metrics_dict['mse'] = mean_squared_error(targets, predictions)
            elif metric.lower() == 'mae':
                metrics_dict['mae'] = mean_absolute_error(targets, predictions)
            elif metric.lower() == 'r2':
                metrics_dict['r2'] = r2_score(targets, predictions)
            elif metric.lower() == 'ccc':
                metrics_dict['ccc'] = self._compute_ccc(targets, predictions)
            elif metric.lower() == 'rmse':
                metrics_dict['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        
        return metrics_dict
    
    def _compute_ccc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Concordance Correlation Coefficient.
        
        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.
            
        Returns:
            CCC value.
        """
        # Pearson correlation coefficient
        cor = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Means and variances
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        
        # CCC formula
        numerator = 2 * cor * np.sqrt(var_true) * np.sqrt(var_pred)
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        
        ccc = numerator / denominator if denominator != 0 else 0
        return ccc
    
    def _save_model(self, save_path: str, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint.
        
        Args:
            save_path: Path to save the model.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint data.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint 