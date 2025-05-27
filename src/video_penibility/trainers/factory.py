"""Trainer factory for creating trainers and optimizers based on configuration."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Type, Tuple
import logging

from .base import BaseTrainer
from .regression_trainer import RegressionTrainer
from ..config.schema import TrainingConfig

logger = logging.getLogger(__name__)


class TrainerFactory:
    """Factory class for creating trainers based on configuration."""
    
    _trainer_registry: Dict[str, Type[BaseTrainer]] = {
        "regression": RegressionTrainer,
        "default": RegressionTrainer,
    }
    
    _optimizer_registry = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
    }
    
    _criterion_registry = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
        "smooth_l1": nn.SmoothL1Loss,
    }
    
    _scheduler_registry = {
        "step": optim.lr_scheduler.StepLR,
        "plateau": optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": optim.lr_scheduler.CosineAnnealingLR,
        "exponential": optim.lr_scheduler.ExponentialLR,
    }
    
    @classmethod
    def register_trainer(cls, name: str, trainer_class: Type[BaseTrainer]) -> None:
        """Register a new trainer class.
        
        Args:
            name: Name to register the trainer under.
            trainer_class: Trainer class to register.
        """
        cls._trainer_registry[name.lower()] = trainer_class
        logger.info(f"Registered trainer: {name} -> {trainer_class.__name__}")
    
    @classmethod
    def create_optimizer(
        cls,
        model: nn.Module,
        training_config: TrainingConfig
    ) -> torch.optim.Optimizer:
        """Create optimizer based on configuration.
        
        Args:
            model: Model to optimize.
            training_config: Training configuration.
            
        Returns:
            Configured optimizer.
        """
        optimizer_name = training_config.optimizer.lower()
        
        if optimizer_name not in cls._optimizer_registry:
            available_optimizers = list(cls._optimizer_registry.keys())
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. "
                f"Available optimizers: {available_optimizers}"
            )
        
        optimizer_class = cls._optimizer_registry[optimizer_name]
        
        # Base optimizer arguments
        optimizer_args = {
            "lr": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
        }
        
        # Add optimizer-specific arguments
        if optimizer_name in ["adam", "adamw"]:
            optimizer_args["betas"] = (0.9, 0.999)
            optimizer_args["eps"] = 1e-8
        elif optimizer_name == "sgd":
            optimizer_args["momentum"] = 0.9
        
        optimizer = optimizer_class(model.parameters(), **optimizer_args)
        logger.info(f"Created {optimizer_name} optimizer with lr={training_config.learning_rate}")
        
        return optimizer
    
    @classmethod
    def create_criterion(cls, training_config: TrainingConfig) -> nn.Module:
        """Create loss criterion based on configuration.
        
        Args:
            training_config: Training configuration.
            
        Returns:
            Loss criterion.
        """
        criterion_name = training_config.loss_function.lower()
        
        if criterion_name not in cls._criterion_registry:
            available_criteria = list(cls._criterion_registry.keys())
            raise ValueError(
                f"Unsupported loss function: {criterion_name}. "
                f"Available criteria: {available_criteria}"
            )
        
        criterion_class = cls._criterion_registry[criterion_name]
        criterion = criterion_class()
        
        logger.info(f"Created {criterion_name} loss criterion")
        return criterion
    
    @classmethod
    def create_scheduler(
        cls,
        optimizer: torch.optim.Optimizer,
        training_config: TrainingConfig
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on configuration.
        
        Args:
            optimizer: Optimizer to schedule.
            training_config: Training configuration.
            
        Returns:
            Learning rate scheduler or None.
        """
        if training_config.scheduler is None:
            return None
        
        scheduler_name = training_config.scheduler.lower()
        
        if scheduler_name not in cls._scheduler_registry:
            available_schedulers = list(cls._scheduler_registry.keys())
            raise ValueError(
                f"Unsupported scheduler: {scheduler_name}. "
                f"Available schedulers: {available_schedulers}"
            )
        
        scheduler_class = cls._scheduler_registry[scheduler_name]
        scheduler_params = training_config.scheduler_params.copy()
        
        # Add default parameters for specific schedulers
        if scheduler_name == "step":
            scheduler_params.setdefault("step_size", 30)
            scheduler_params.setdefault("gamma", 0.1)
        elif scheduler_name == "plateau":
            scheduler_params.setdefault("mode", "min")
            scheduler_params.setdefault("factor", 0.5)
            scheduler_params.setdefault("patience", 10)
        elif scheduler_name == "cosine":
            scheduler_params.setdefault("T_max", 50)
        elif scheduler_name == "exponential":
            scheduler_params.setdefault("gamma", 0.95)
        
        scheduler = scheduler_class(optimizer, **scheduler_params)
        logger.info(f"Created {scheduler_name} scheduler with params: {scheduler_params}")
        
        return scheduler
    
    @classmethod
    def create_trainer(
        cls,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config: TrainingConfig,
        device: torch.device,
        trainer_type: str = "regression"
    ) -> BaseTrainer:
        """Create trainer based on configuration.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            training_config: Training configuration.
            device: Device to train on.
            trainer_type: Type of trainer to create.
            
        Returns:
            Configured trainer.
        """
        trainer_name = trainer_type.lower()
        
        if trainer_name not in cls._trainer_registry:
            available_trainers = list(cls._trainer_registry.keys())
            raise ValueError(
                f"Unsupported trainer type: {trainer_name}. "
                f"Available trainers: {available_trainers}"
            )
        
        # Create optimizer, criterion, and scheduler
        optimizer = cls.create_optimizer(model, training_config)
        criterion = cls.create_criterion(training_config)
        scheduler = cls.create_scheduler(optimizer, training_config)
        
        # Create trainer
        trainer_class = cls._trainer_registry[trainer_name]
        trainer = trainer_class(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            metrics=training_config.metrics,
            early_stopping_patience=training_config.early_stopping_patience,
        )
        
        logger.info(f"Created {trainer_name} trainer")
        return trainer
    
    @classmethod
    def list_available_trainers(cls) -> list:
        """List all available trainer types.
        
        Returns:
            List of available trainer names.
        """
        return list(cls._trainer_registry.keys())
    
    @classmethod
    def list_available_optimizers(cls) -> list:
        """List all available optimizer types.
        
        Returns:
            List of available optimizer names.
        """
        return list(cls._optimizer_registry.keys())
    
    @classmethod
    def list_available_criteria(cls) -> list:
        """List all available loss criteria.
        
        Returns:
            List of available criterion names.
        """
        return list(cls._criterion_registry.keys())
    
    @classmethod
    def list_available_schedulers(cls) -> list:
        """List all available scheduler types.
        
        Returns:
            List of available scheduler names.
        """
        return list(cls._scheduler_registry.keys()) 