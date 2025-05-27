# 🎬 Video Penibility Assessment Framework

A comprehensive deep learning framework for assessing physical penibility (strain/difficulty) in videos using multiple feature extraction methods and temporal modeling approaches.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Enabled-orange.svg)](https://www.tensorflow.org/tensorboard)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 **Key Features**

### **🧠 Multiple Feature Extraction Methods**
- **I3D**: Inflated 3D ConvNets for video understanding
- **Swin3D**: 3D Swin Transformer variants (Tiny, Small, Base)
- **FaceNet**: Face-based feature extraction
- **FaceMesh**: 3D facial landmark detection
- **LibreFace**: Advanced facial analysis features

### **🎯 Advanced Temporal Modeling**
- **Transformer**: Multi-head attention for sequence modeling
- **TCN**: Temporal Convolutional Networks
- **LSTM/GRU**: Recurrent neural networks for temporal dependencies

### **📊 Comprehensive Monitoring & Visualization**
- **TensorBoard Integration**: Real-time training monitoring
- **Prediction vs Target Plots**: Visual assessment of model performance
- **Rich Console Output**: Beautiful training progress with Rich library
- **Cross-Validation Metrics**: MSE, MAE, R², CCC with statistical analysis

### **⚙️ Robust Training Framework**
- **Subject-wise Cross-Validation**: Proper evaluation avoiding data leakage
- **Configurable YAML**: Easy experiment configuration and reproduction
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Model Checkpointing**: Save best models during training
- **Gradient Clipping**: Stable training for sequence models

## 📁 **Project Structure**

```
video_penibility_assessment/
├── 📁 src/video_penibility/           # Core framework
│   ├── 📁 config/                     # Configuration management
│   ├── 📁 datasets/                   # Dataset loading and processing
│   ├── 📁 models/                     # Model architectures
│   ├── 📁 training/                   # Training framework
│   └── 📁 utils/                      # Utility functions
├── 📁 scripts/                        # Training and evaluation scripts
├── 📁 configs/                        # Experiment configurations
├── 📁 results/                        # Training outputs and logs
├── 📄 requirements.txt                # Python dependencies
└── 📄 README.md                       # This file
```

## 🛠️ **Installation**

### **Prerequisites**
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for large feature sets

### **Setup**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd video_penibility_assessment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv ai_vision
   source ai_vision/bin/activate  # Linux/Mac
   # or
   ai_vision\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 **Quick Start**

### **1. Basic Training Example**

```bash
# Train I3D + TCN model
python scripts/train.py --config configs/i3d_tcn.yaml

# Train Swin3D + Transformer model
python scripts/train.py --config configs/swin3d_transformer_custom.yaml
```

### **2. Monitor Training with TensorBoard**

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir=results/

# View at: http://localhost:6006
```

### **3. Create Custom Configuration**

```yaml
# configs/my_experiment.yaml
experiment:
  name: "my_experiment"
  device: "cuda:0"
  output_dir: "results/my_experiment"

data:
  features_type: "i3d"                    # i3d, swin3d_t, facenet, etc.
  data_type: "full_body"
  annotation_path: "/path/to/annotations.csv"
  features_root: "/path/to/features/"

model:
  name: "transformer"                     # transformer, tcn, lstm, gru
  hidden_dim: 512
  num_layers: 4
  dropout: 0.2

training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 150
  optimizer: "adamw"
```

## 📊 **Experiment Results**

### **Current Best Performance**

| **Features** | **Model** | **MSE** | **R²** | **CCC** | **Status** |
|-------------|-----------|---------|--------|---------|------------|
| I3D         | GRU       | 6.87±4.71 | 0.20   | **0.59** | ✅ Best |
| I3D         | TCN       | ~3.8    | ~0.5   | ~0.7    | 🔄 Current |
| Swin3D-T    | Transformer | 10.07±1.25 | -0.22 | -0.004 | ❌ Poor |

### **Features Comparison**
- **I3D (1024-dim)**: Better for motion understanding
- **Swin3D-T (768-dim)**: More compact but lower performance
- **Cross-validation**: 5-fold subject-wise splitting

## 🔧 **Configuration Options**

### **Supported Features**
```yaml
features_type:
  - "i3d"              # Inflated 3D ConvNets
  - "swin3d_t"         # Swin3D Tiny
  - "swin3d_s"         # Swin3D Small  
  - "swin3d_b"         # Swin3D Base
  - "facenet"          # Face features
  - "facemesh"         # Facial landmarks
  - "libreface"        # Advanced face analysis
```

### **Supported Models**
```yaml
model_name:
  - "transformer"      # Multi-head attention
  - "tcn"             # Temporal Convolutional Network
  - "lstm"            # Long Short-Term Memory
  - "gru"             # Gated Recurrent Unit
```

### **Training Options**
```yaml
training:
  optimizer: ["adam", "adamw", "sgd"]
  loss_function: ["mse", "mae", "huber"]
  scheduler: ["step", "plateau", null]
  early_stopping_patience: 30
```

## 📈 **TensorBoard Visualizations**

The framework provides comprehensive logging:

- **📊 Training/Validation Metrics**: Loss, MSE, MAE, R², CCC
- **🎯 Prediction Plots**: Scatter plots with residuals analysis
- **📈 Learning Curves**: Training progress over epochs
- **🔍 Model Architecture**: Network graph visualization
- **📊 Cross-Validation Summary**: Statistical analysis across folds

## 🧪 **Experiment Management**

### **Configuration-Based Experiments**
- All experiments defined in YAML files
- Easy parameter sweeps and ablation studies
- Automatic result organization by experiment name

### **Reproducibility**
- Fixed random seeds across experiments
- Version control for configurations
- Complete logging of hyperparameters

### **Cross-Validation**
- Subject-wise splitting (no data leakage)
- Statistical significance testing
- Variance analysis across folds

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- PyTorch team for the excellent deep learning framework
- TensorBoard for comprehensive experiment visualization
- Rich library for beautiful console output
- Contributors to the various feature extraction methods

## 📚 **Citation**

If you use this framework in your research, please cite:

```bibtex
@software{video_penibility_framework,
  title={Video Penibility Assessment Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/video_penibility_assessment}
}
```

---

**🎯 Ready to assess video penibility with state-of-the-art deep learning!**
