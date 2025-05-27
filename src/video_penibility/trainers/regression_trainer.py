"""Regression trainer implementation."""

import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np
import logging

from .base import BaseTrainer

logger = logging.getLogger(__name__)


class RegressionTrainer(BaseTrainer):
    """Trainer for regression tasks."""
    
    def __init__(self, **kwargs):
        """Initialize regression trainer.
        
        Args:
            **kwargs: Arguments passed to BaseTrainer.
        """
        super().__init__(**kwargs)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        for batch_idx, (features, targets) in enumerate(self.train_loader):
            # Move data to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(features)
            predictions = predictions.squeeze(-1)  # Remove last dimension if needed
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Store predictions and targets for metrics computation
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
            num_batches += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        
        # Convert to numpy arrays
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        
        # Compute additional metrics
        metrics = self.compute_metrics(predictions_np, targets_np)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(self.val_loader):
                # Move data to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(features)
                predictions = predictions.squeeze(-1)  # Remove last dimension if needed
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                
                # Accumulate metrics
                total_loss += loss.item()
                
                # Store predictions and targets for metrics computation
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                num_batches += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        
        # Convert to numpy arrays
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        
        # Compute additional metrics
        metrics = self.compute_metrics(predictions_np, targets_np)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def predict(self, data_loader) -> np.ndarray:
        """Make predictions on a dataset.
        
        Args:
            data_loader: DataLoader with data to predict on.
            
        Returns:
            Array of predictions.
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for features, _ in data_loader:
                features = features.to(self.device)
                predictions = self.model(features)
                predictions = predictions.squeeze(-1)
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions)
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader with data to evaluate on.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                predictions = predictions.squeeze(-1)
                
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                num_batches += 1
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        predictions_np = np.array(all_predictions)
        targets_np = np.array(all_targets)
        
        metrics = self.compute_metrics(predictions_np, targets_np)
        metrics['loss'] = avg_loss
        
        return metrics 