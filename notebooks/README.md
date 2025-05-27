# Notebooks

This directory contains Jupyter notebooks for exploring and demonstrating the Video Penibility Assessment framework.

## Available Notebooks

- `01_getting_started.ipynb` - Introduction to the framework and basic usage
- `02_model_comparison.ipynb` - Comparing different model architectures
- `03_feature_analysis.ipynb` - Analyzing different feature types
- `04_hyperparameter_tuning.ipynb` - Hyperparameter optimization examples
- `05_results_visualization.ipynb` - Visualizing training results and metrics

## Setup

Before running the notebooks, ensure you have installed the package and its dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

Start Jupyter Lab or Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Navigate to the notebooks directory and open the desired notebook.

## Data Requirements

Some notebooks may require sample data. Please ensure you have:

- Annotation files in `data/labels/`
- Pre-extracted features in `data/features/`
- Proper configuration files in `configs/`

See the main README for data preparation instructions. 