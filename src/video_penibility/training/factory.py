"""Enhanced trainer factory with TensorBoard and Rich integration."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from ..config.schema import TrainingConfig, ModelConfig
from ..models import ModelFactory

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Custom collate function to handle variable sequence lengths."""
    features_list = []
    targets_list = []
    
    for features, target in batch:
        features_list.append(features)
        targets_list.append(target)
    
    # Pad sequences to the same length
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    targets = torch.stack(targets_list)
    
    return padded_features, targets


def create_prediction_plot(targets: np.ndarray, predictions: np.ndarray, fold: int, epoch: int, phase: str = "val") -> plt.Figure:
    """Create prediction vs target scatter plot with residuals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(targets, predictions, alpha=0.6, s=30, c='blue')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'Fold {fold} - Epoch {epoch} ({phase.title()})\nPredictions vs True Values')
    ax1.grid(True, alpha=0.3)
    
    # Add metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Residuals plot
    residuals = predictions - targets
    ax2.scatter(targets, residuals, alpha=0.6, s=30, c='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Residuals (Pred - True)')
    ax2.set_title(f'Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    # Add residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax2.text(0.05, 0.95, f'Mean = {mean_residual:.3f}\nStd = {std_residual:.3f}', 
             transform=ax2.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    return fig


class EnhancedTrainer:
    """Enhanced trainer with TensorBoard and Rich integration."""
    
    def __init__(
        self,
        training_config: TrainingConfig,
        feature_dim: int,
        model_config: ModelConfig,
        device: str
    ):
        """Initialize the enhanced trainer.
        
        Args:
            training_config: Training configuration.
            feature_dim: Input feature dimension.
            model_config: Model configuration.
            device: Device to train on.
        """
        self.training_config = training_config
        self.model_config = model_config
        self.device = torch.device(device)
        self.feature_dim = feature_dim
        
        # Create model
        self.model = ModelFactory.create_model(model_config, feature_dim)
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create loss function
        self.criterion = self._create_criterion()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.training_config.optimizer.lower()
        
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss criterion based on configuration."""
        loss_name = self.training_config.loss_function.lower()
        
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae":
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if self.training_config.scheduler is None:
            return None
        
        scheduler_name = self.training_config.scheduler.lower()
        
        if scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.scheduler_params.get("step_size", 30),
                gamma=self.training_config.scheduler_params.get("gamma", 0.1)
            )
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.training_config.scheduler_params.get("factor", 0.5),
                patience=self.training_config.scheduler_params.get("patience", 10)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def train_fold(
        self,
        train_subset: Subset,
        val_subset: Subset,
        fold_num: int,
        output_dir: str,
        writer: Optional[SummaryWriter] = None
    ) -> Dict[str, float]:
        """Train one fold with TensorBoard logging.
        
        Args:
            train_subset: Training data subset.
            val_subset: Validation data subset.
            fold_num: Fold number.
            output_dir: Output directory.
            writer: TensorBoard writer for logging.
            
        Returns:
            Dictionary with final metrics.
        """
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        # Progress tracking with Rich
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            
            epoch_task = progress.add_task(
                f"   Training", 
                total=self.training_config.num_epochs
            )
            
            for epoch in range(self.training_config.num_epochs):
                # Training phase
                train_metrics, train_preds, train_targets, train_step = self._train_epoch(
                    train_loader, epoch, fold_num, writer, global_step
                )
                global_step = train_step
                
                # Validation phase
                val_metrics, val_preds, val_targets = self._validate_epoch(val_loader)
                
                # Log to TensorBoard
                if writer is not None:
                    # Log metrics
                    for metric, value in train_metrics.items():
                        writer.add_scalar(f'fold_{fold_num}/train/{metric}', value, epoch)
                    
                    for metric, value in val_metrics.items():
                        writer.add_scalar(f'fold_{fold_num}/val/{metric}', value, epoch)
                    
                    # Log learning rate
                    writer.add_scalar(f'fold_{fold_num}/learning_rate', 
                                    self.optimizer.param_groups[0]['lr'], epoch)
                    
                    # Log prediction plots every 10 epochs or at the end
                    if epoch % 10 == 0 or epoch == self.training_config.num_epochs - 1:
                        # Training predictions plot
                        train_fig = create_prediction_plot(train_targets, train_preds, 
                                                         fold_num, epoch, "train")
                        writer.add_figure(f'fold_{fold_num}/train_predictions', train_fig, epoch)
                        plt.close(train_fig)
                        
                        # Validation predictions plot
                        val_fig = create_prediction_plot(val_targets, val_preds, 
                                                       fold_num, epoch, "val")
                        writer.add_figure(f'fold_{fold_num}/val_predictions', val_fig, epoch)
                        plt.close(val_fig)
                        
                        # Log prediction histograms
                        writer.add_histogram(f'fold_{fold_num}/train_predictions_hist', 
                                           train_preds, epoch)
                        writer.add_histogram(f'fold_{fold_num}/val_predictions_hist', 
                                           val_preds, epoch)
                        writer.add_histogram(f'fold_{fold_num}/train_targets_hist', 
                                           train_targets, epoch)
                        writer.add_histogram(f'fold_{fold_num}/val_targets_hist', 
                                           val_targets, epoch)
                
                # Update progress
                progress.update(
                    epoch_task,
                    advance=1,
                    description=f"   Epoch {epoch+1:3d} - Loss: {val_metrics['loss']:.3f}"
                )
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    # Save best model
                    if self.training_config.save_best_model:
                        save_path = Path(output_dir) / f"fold_{fold_num}_best_model.pth"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_val_loss': best_val_loss,
                            'val_metrics': val_metrics,
                        }, save_path)
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= self.training_config.early_stopping_patience:
                    progress.update(epoch_task, description=f"   Early stopped at epoch {epoch+1}")
                    break
        
        return {
            "val_loss": best_val_loss,
            "val_mse": val_metrics.get('mse', 0),
            "val_mae": val_metrics.get('mae', 0),
            "val_r2": val_metrics.get('r2', 0),
            "val_ccc": val_metrics.get('ccc', 0),
        }
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int, fold: int, 
                     writer: Optional[SummaryWriter], global_step: int) -> tuple:
        """Train for one epoch with detailed logging."""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(features)
            predictions = predictions.squeeze(-1)
            
            loss = self.criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
            # Log batch metrics to TensorBoard (every 10 batches)
            if writer is not None and batch_idx % 10 == 0:
                writer.add_scalar(f'fold_{fold}/train_batch/loss', loss.item(), global_step)
                
                # Log gradient norms
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                writer.add_scalar(f'fold_{fold}/train_batch/grad_norm', total_norm, global_step)
            
            global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics, np.array(all_predictions), np.array(all_targets), global_step
    
    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                predictions = predictions.squeeze(-1)
                
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics, np.array(all_predictions), np.array(all_targets)
    
    def _compute_metrics(self, predictions, targets) -> Dict[str, float]:
        """Compute evaluation metrics."""
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # Concordance Correlation Coefficient
        metrics['ccc'] = self._compute_ccc(targets, predictions)
        
        return metrics
    
    def _compute_ccc(self, y_true, y_pred) -> float:
        """Compute Concordance Correlation Coefficient."""
        import numpy as np
        
        # Pearson correlation coefficient
        cor = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(cor):
            return 0.0
        
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


class TrainerFactory:
    """Factory class for creating enhanced trainers."""
    
    @classmethod
    def create_trainer(
        cls,
        training_config: TrainingConfig,
        feature_dim: int,
        model_config: ModelConfig,
        device: str
    ) -> EnhancedTrainer:
        """Create an enhanced trainer.
        
        Args:
            training_config: Training configuration.
            feature_dim: Input feature dimension.
            model_config: Model configuration.
            device: Device to train on.
            
        Returns:
            Enhanced trainer instance.
        """
        return EnhancedTrainer(training_config, feature_dim, model_config, device) 