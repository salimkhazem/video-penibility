#!/usr/bin/env python3
"""Enhanced training script with TensorBoard and Rich integration."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
import yaml

from video_penibility.config.schema import MainConfig, load_config
from video_penibility.datasets import DatasetFactory
from video_penibility.models import ModelFactory
from video_penibility.training import TrainerFactory
from video_penibility.utils.cross_validation import create_subject_splits


# Initialize Rich console
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging with Rich handler."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def print_experiment_header(config: MainConfig, config_file: str) -> None:
    """Print beautiful experiment header with Rich."""
    # Create experiment info table
    table = Table(title="🚀 Experiment Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    table.add_row("Experiment Name", config.experiment.name)
    table.add_row("Config File", config_file)
    table.add_row("Device", config.experiment.device)
    table.add_row("Output Directory", config.experiment.output_dir)
    table.add_row("Random Seed", str(config.experiment.seed))
    table.add_row("Tags", ", ".join(config.experiment.tags))
    
    console.print(table)
    console.print()


def print_data_info(config: MainConfig, dataset_size: int, feature_dim: int) -> None:
    """Print data configuration information."""
    table = Table(title="📊 Data Configuration", show_header=True, header_style="bold blue")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    table.add_row("Features Type", config.data.features_type)
    table.add_row("Data Type", config.data.data_type)
    table.add_row("Dataset Size", str(dataset_size))
    table.add_row("Feature Dimension", str(feature_dim))
    table.add_row("Cross-validation Splits", str(config.cross_validation.n_splits))
    table.add_row("CV Strategy", config.cross_validation.strategy)
    
    console.print(table)
    console.print()


def print_model_info(config: MainConfig, model: torch.nn.Module) -> None:
    """Print model architecture information once."""
    table = Table(title="🧠 Model Architecture", show_header=True, header_style="bold green")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    table.add_row("Model Type", config.model.name.upper())
    table.add_row("Hidden Dimension", str(config.model.hidden_dim))
    table.add_row("Number of Layers", str(config.model.num_layers))
    table.add_row("Dropout", str(config.model.dropout))
    table.add_row("Output Dimension", str(config.model.output_dim))
    
    # Add model-specific parameters
    if hasattr(config.model, 'extra_params') and config.model.extra_params:
        for key, value in config.model.extra_params.items():
            table.add_row(f"Extra: {key}", str(value))
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    
    console.print(table)
    console.print()


def print_training_info(config: MainConfig) -> None:
    """Print training configuration."""
    table = Table(title="⚙️ Training Configuration", show_header=True, header_style="bold yellow")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    table.add_row("Batch Size", str(config.training.batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Weight Decay", str(config.training.weight_decay))
    table.add_row("Number of Epochs", str(config.training.num_epochs))
    table.add_row("Optimizer", config.training.optimizer)
    table.add_row("Loss Function", config.training.loss_function)
    table.add_row("Early Stopping Patience", str(config.training.early_stopping_patience))
    
    console.print(table)
    console.print()


def initialize_tensorboard(config: MainConfig, config_file: str) -> SummaryWriter:
    """Initialize TensorBoard with experiment tracking."""
    # Create run name with config file info
    run_name = f"{config.experiment.name}_{Path(config_file).stem}"
    log_dir = Path(config.experiment.output_dir) / "tensorboard" / run_name
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Log experiment configuration
    config_text = f"""
    **Experiment Configuration**
    - Name: {config.experiment.name}
    - Config File: {config_file}
    - Features: {config.data.features_type}
    - Model: {config.model.name}
    - Data Type: {config.data.data_type}
    - Batch Size: {config.training.batch_size}
    - Learning Rate: {config.training.learning_rate}
    - Epochs: {config.training.num_epochs}
    - Device: {config.experiment.device}
    - Tags: {', '.join(config.experiment.tags)}
    """
    
    writer.add_text("experiment/config", config_text, 0)
    
    console.print(f"📊 [green]TensorBoard initialized at: {log_dir}[/green]")
    console.print(f"🌐 [blue]Run: tensorboard --logdir={log_dir.parent}[/blue]")
    
    return writer


def create_prediction_plot(targets: np.ndarray, predictions: np.ndarray, fold: int, epoch: int) -> plt.Figure:
    """Create prediction vs target scatter plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(targets, predictions, alpha=0.6, s=30)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title(f'Fold {fold} - Epoch {epoch}\nPredictions vs True Values')
    ax1.grid(True, alpha=0.3)
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(targets, predictions)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Residuals plot
    residuals = predictions - targets
    ax2.scatter(targets, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Residuals (Pred - True)')
    ax2.set_title(f'Residuals Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_cross_validation(config: MainConfig, config_file: str) -> Dict[str, Any]:
    """Run cross-validation training with TensorBoard logging."""
    console.print(Panel.fit("🔄 Starting Cross-Validation Training", style="bold white on blue"))
    
    # Initialize TensorBoard
    writer = initialize_tensorboard(config, config_file)
    
    # Create dataset
    with console.status("[bold green]Loading dataset..."):
        dataset = DatasetFactory.create_dataset(config.data)
        feature_dim = dataset.get_feature_dim()
    
    # Print all configuration info once
    print_data_info(config, len(dataset), feature_dim)
    
    # Create model to show architecture
    model = ModelFactory.create_model(config.model, feature_dim)
    print_model_info(config, model)
    print_training_info(config)
    
    # Log model architecture to TensorBoard
    try:
        # Create dummy input for model graph
        dummy_input = torch.randn(1, 32, feature_dim)  # (batch, seq_len, features)
        writer.add_graph(model, dummy_input)
        console.print("📈 [green]Model architecture logged to TensorBoard[/green]")
    except Exception as e:
        console.print(f"⚠️ [yellow]Could not log model graph: {e}[/yellow]")
    
    # Create cross-validation splits
    console.print("📋 Creating cross-validation splits...")
    splits = create_subject_splits(dataset, config.cross_validation.n_splits, config.cross_validation.random_state)
    
    # Results storage
    all_results = []
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        cv_task = progress.add_task("Cross-Validation Progress", total=config.cross_validation.n_splits)
        
        for fold, (train_indices, val_indices) in enumerate(splits):
            fold_description = f"Fold {fold + 1}/{config.cross_validation.n_splits}"
            progress.update(cv_task, description=f"Training {fold_description}")
            
            console.print(f"\n🔹 [bold cyan]{fold_description}[/bold cyan]")
            console.print(f"   Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
            
            # Create data subsets
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            # Create trainer
            trainer = TrainerFactory.create_trainer(
                config.training, 
                feature_dim, 
                config.model, 
                config.experiment.device
            )
            
            # Train with TensorBoard logging
            fold_results = trainer.train_fold(
                train_subset, 
                val_subset, 
                fold + 1,
                config.experiment.output_dir,
                writer=writer
            )
            
            all_results.append(fold_results)
            
            # Log fold summary to TensorBoard
            writer.add_scalars(f'fold_summary/fold_{fold+1}', {
                'val_mse': fold_results["val_mse"],
                'val_mae': fold_results["val_mae"],
                'val_r2': fold_results["val_r2"],
                'val_ccc': fold_results["val_ccc"],
            }, fold)
            
            # Brief fold summary
            console.print(f"   ✅ MSE: {fold_results['val_mse']:.3f}, R²: {fold_results['val_r2']:.3f}, CCC: {fold_results['val_ccc']:.3f}")
            
            progress.advance(cv_task)
    
    # Log final cross-validation statistics
    metrics = ["val_mse", "val_mae", "val_r2", "val_ccc"]
    for metric in metrics:
        values = [fold[metric] for fold in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        writer.add_scalar(f'cv_summary/{metric}_mean', mean_val, 0)
        writer.add_scalar(f'cv_summary/{metric}_std', std_val, 0)
        
        # Create histogram of metric values across folds
        writer.add_histogram(f'cv_distribution/{metric}', np.array(values), 0)
    
    # Close TensorBoard writer
    writer.close()
    
    return {"folds": all_results, "config_file": config_file}


def print_final_results(results: Dict[str, Any]) -> None:
    """Print beautiful final results summary."""
    fold_results = results["folds"]
    config_file = results["config_file"]
    
    # Calculate statistics
    metrics = ["val_mse", "val_mae", "val_r2", "val_ccc"]
    stats = {}
    
    for metric in metrics:
        values = [fold[metric] for fold in fold_results]
        stats[metric] = {
            "mean": sum(values) / len(values),
            "std": (sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5
        }
    
    # Create results table
    table = Table(title="🎯 Final Cross-Validation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean ± Std", style="white")
    table.add_column("Individual Folds", style="dim white")
    
    for metric in metrics:
        metric_name = metric.replace("val_", "").upper()
        mean_std = f"{stats[metric]['mean']:.3f} ± {stats[metric]['std']:.3f}"
        fold_values = ", ".join([f"{fold[metric]:.3f}" for fold in fold_results])
        table.add_row(metric_name, mean_std, fold_values)
    
    console.print()
    console.print(table)
    
    # Success message
    console.print()
    console.print(Panel.fit(
        f"✨ Experiment completed successfully!\nConfig: {config_file}\nTensorBoard logs saved locally",
        style="bold white on green"
    ))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train video penibility assessment model")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config.experiment.log_level)
    
    # Print experiment header
    print_experiment_header(config, args.config)
    
    # Set random seed
    torch.manual_seed(config.experiment.seed)
    
    # Create output directory
    Path(config.experiment.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Run training
        results = run_cross_validation(config, args.config)
        
        # Print and save results
        print_final_results(results)
        
        # Save results
        results_file = Path(config.experiment.output_dir) / "results.json"
        with open(results_file, "w") as f:
            # Use custom encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, 'item'):
                        return obj.item()
                    return super().default(obj)
            
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        console.print(f"📄 Results saved to: {results_file}")
        console.print(f"📊 [blue]View TensorBoard: tensorboard --logdir={Path(config.experiment.output_dir) / 'tensorboard'}[/blue]")
        
    except KeyboardInterrupt:
        console.print("\n⚠️ Training interrupted by user", style="bold yellow")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Training failed: {str(e)}", style="bold red")
        raise


if __name__ == "__main__":
    main() 