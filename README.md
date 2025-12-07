# TabularTextMultimodalFusion

A unified framework for experimenting with various architectures that combine **tabular data** (numerical + categorical) and **textual data** using pretrained language models (e.g., BERT/DistilBERT).  

Inspired by and extending the ideas in [TabularTextTransformer](https://github.com/yury-petyushin/TabularTextTransformer), this repo explores fusion architectures, contrastive learning, and graph-based methods for multimodal classification.

---

## 🔧 Features

- **Multimodal Fusion**: Cross-attention, skip connections, late fusion, GAT-based fusion
- **Advanced Encodings**: Custom numerical encodings (RBF, Fourier, Chebyshev, Sigmoid, Positional vectors)
- **Graph Neural Networks**: Graph-based multimodal GNN via `torch_geometric`
- **Contrastive Learning**: Multiple contrastive loss variants (MMD, MINE, InfoNCE)
- **Comprehensive Benchmarking**: Multiple datasets with standardized preprocessing
- **Unified Framework**: Consistent API for all model architectures

---

## 📦 Installation

### Option 1: Install from PyPI (Recommended for Users)

For users who want to use the package without modifying the source code:

```bash
# 1. Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install torch-geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 3. Install the package from PyPI
pip install tabulartextmultimodalfusion

# 4. Install additional dependencies
pip install transformers pandas numpy scikit-learn scipy matplotlib seaborn optuna omegaconf
```

**Quick Start:**

```python
from tabulartextmultimodalfusion.models import CrossAttention, CombinedModelConcat4, TabularEmbedding
from tabulartextmultimodalfusion.dataset import prepareTensorDatasetWithTokenizer, preprocess_dataset
from tabulartextmultimodalfusion.settings import load_settings

# See example.py for complete usage
```

---

### Option 2: Development Installation (For Contributors/Researchers)

For development, experiments, or contributing to the project:

#### Method A: Using Conda (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/nadav22799/TabularTextMultimodalFusion
cd TabularTextMultimodalFusion

# 2. Create conda environment from environment.yaml
conda env create -f environment.yaml

# 3. Activate the environment
conda activate TTMF

# 4. Install remaining dependencies and package in editable mode
pip install -r requirements.txt && pip install -e .

# 5. Run example to verify installation
python example.py
```

#### Method B: Using pip only

```bash
# 1. Clone the repository
git clone https://github.com/nadav22799/TabularTextMultimodalFusion
cd TabularTextMultimodalFusion

# 2. Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install package in editable mode
pip install -e .

# 5. Run example to verify installation
python example.py
```

**Note:** See [`example.py`](example.py) for a complete working example with synthetic data, model initialization, and training.

---

## 🎯 Model Selection Guide

### Base Model Architectures

Choose from the following model families based on your use case:

#### 🔥 Cross-Attention Models (Recommended)
Our proposed approaches for optimal text-tabular fusion:
- **`CrossAttention`**: Core cross-attention mechanism between text and tabular features
- **`CrossAttentionSkipNet`**: Cross-attention enhanced with skip connections for better gradient flow

#### 🔄 Fusion-Based Models
Alternative fusion strategies:
- **`FusionSkipNet`**: Skip connections with feature fusion
- **`CombinedModelGAT`**: Graph Attention Network for combined feature processing

#### 🤖 BERT-Based Approaches
Different strategies for incorporating BERT:
- **`LateFuseBERT`**: Late fusion of BERT text embeddings with tabular features
- **`AllTextBERT`**: Converts tabular data to text for unified BERT processing
- **`TabularForBert`**: Tabular data preprocessing optimized for BERT compatibility
- **`BertWithTabular`**: BERT with additional tabular feature processing layers

#### 📊 Single-Modality Baselines
For comparison and ablation studies:
- **`OnlyTabular`**: Tabular data only (MLP-based)
- **`OnlyText`**: Text data only (BERT-based)

### Configuration Options

#### Fusion Methods
Control how text and tabular features are combined:

```python
# Without BERT self-attention on final embeddings
fusion_methods = ['Concat2', 'Concat4', 'SumW2', 'SumW4']

# With BERT self-attention on final embeddings (suffix 's')
fusion_methods = ['Concat2s', 'Concat4s', 'SumW2s', 'SumW4s']
```

- **Concat**: Concatenation fusion (2 = 2x dims, 4 = 4x dims)
- **SumW**: Weighted sum fusion (2 = 2x dims, 4 = 4x dims)
- **'s' suffix**: Applies BERT self-attention on final token embeddings

#### Numerical Encoders
Transform numerical tabular features for better cross-modal alignment:

- **`Fourier`**: Fourier feature encoding for periodic patterns
- **`FourierVec`**: Vectorized Fourier encoding
- **`PosEnVec`**: Positional encoding vectors
- **`RBF`**: Radial Basis Function encoding for non-linear relationships
- **`RBFVec`**: Vectorized RBF encoding
- **`Sigmoid`**: Sigmoid transformation for bounded features
- **`Chebyshev`**: Chebyshev polynomial encoding

#### Loss Functions
Optimize cross-modal representation learning:

- **`MMD`**: Maximum Mean Discrepancy for distribution alignment
- **`MINE`**: Mutual Information Neural Estimation
- **`InfoNCE`**: Info Noise Contrastive Estimation
- **`Contrastive`**: Standard contrastive learning loss

### Model Naming Convention

Models follow the pattern: `{BaseModel}{FusionMethod}[{NumericalEncoder}][{LossFunction}]`

**Examples:**
- `CrossAttentionConcat4`: Cross-attention with 4D concatenation fusion
- `CrossAttentionConcat4s`: Same as above but with self-attention on final embeddings  
- `CrossAttentionConcat4Fourier`: Cross-attention + Concat4 + Fourier encoding
- `CrossAttentionConcat4MMD`: Cross-attention + Concat4 + MMD loss

### 💡 Quick Start Recommendations

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Best Overall Performance** | `CrossAttentionConcat4s` | Optimal fusion with self-attention |
| **Limited Compute** | `CrossAttentionConcat2` | Smaller feature dimensions |
| **Periodic/Seasonal Data** | `CrossAttentionConcat4Fourier` | Fourier encoding for patterns |
| **High-Dimensional Tabular** | `CrossAttentionConcat4RBF` | RBF handles complex relationships |
| **Distribution Alignment** | `CrossAttentionConcat4MMD` | MMD loss for better alignment |
| **Baseline Comparison** | `OnlyText`, `OnlyTabular` | Single-modality benchmarks |

---

## 🚀 Running Experiments

### Quick Start

Run experiments using CLI arguments:

```bash
# Run experiment 1 (architecture comparison)
python src/main.py --version exp1

# Run experiment 2 (numerical encoders)
python src/main.py --version exp2

# Run experiment 3 (loss functions)
python src/main.py --version exp3

# Run MIMIC experiments
python src/main_mimic.py --version exp1
```

### Customization Options

#### 1. **Version Selection** (CLI)
Choose experiment type via command line:
```bash
python src/main.py --version exp1  # Architecture comparison
python src/main.py --version exp2  # Numerical encoder comparison
python src/main.py --version exp3  # Loss function comparison
```

#### 2. **Dataset Selection** (Manual)
Edit `src/main.py` to customize datasets:
```python
DATASETS = ["wine_10", "airbnb", "kick"]  # Select from supported datasets
```

#### 3. **Model Selection** (Automatic by Version)
Models are automatically selected based on the version:
- **`exp1`**: Tests all architecture variants and baselines
- **`exp2`**: Tests numerical encoders with `CrossAttentionConcat4`
- **`exp3`**: Tests loss functions with `CrossAttentionConcat4`

Or manually override in `src/main.py`:
```python
if args.version == "exp1":
    MODELS = ["CrossAttentionConcat4s", "BertWithTabular"]  # Custom selection
```

### 🧪 Experiment Types

| Version | Focus | Models Compared |
|---------|-------|-----------------|
| **`exp1`** | **Architecture Comparison** | All fusion architectures vs baselines |
| **`exp2`** | **Numerical Encoders** | Different encoders with best architecture |
| **`exp3`** | **Loss Functions** | Contrastive learning variants |

#### Experiment 1: Architecture Comparison
Tests fundamental fusion approaches:
```python
MODELS = [
    # Our proposed methods
    "CrossAttentionSumW4", "CrossAttentionConcat4", 
    "CrossAttentionConcat4s", "CrossAttentionSumW4s",
    # Alternative fusion
    "FusionSkipNet", "CombinedModelGAT",
    # BERT variants
    "BertWithTabular", "LateFuseBERT", "AllTextBERT",
    # Baselines
    "OnlyTabular", "OnlyText"
]
```

#### Experiment 2: Numerical Encoder Ablation
Uses best architecture (`CrossAttentionConcat4`) with different encoders:
```python
MODELS = [
    "CrossAttentionConcat4Fourier", "CrossAttentionConcat4RBF",
    "CrossAttentionConcat4FourierVec", "CrossAttentionConcat4PosEnVec",
    "CrossAttentionConcat4Chebyshev", "CrossAttentionConcat4Sigmoid"
]
```

#### Experiment 3: Loss Function Comparison
Tests contrastive learning approaches:
```python
MODELS = [
    "CrossAttentionConcat4MMD", "CrossAttentionConcat4MINE",
    "CrossAttentionConcat4InfoNCE", "CrossAttentionConcat4Contrastive"
]
```

---

## 📁 Package Structure

```
TabularTextMultimodalFusion/
├── src/
│   ├── tabulartextmultimodalfusion/    # Main package
│   │   ├── __init__.py
│   │   ├── models.py                   # Model architectures
│   │   ├── dataset.py                  # Data loading and preprocessing
│   │   ├── settings.py                 # Configuration
│   │   ├── optimization.py             # Training and optimization
│   │   ├── GridSearch.py               # Hyperparameter tuning
│   │   ├── load_mimic.py               # MIMIC dataset loader
│   │   └── mimic_utils.py              # MIMIC utilities
│   ├── main.py                         # Experiment runner
│   ├── main_mimic.py                   # MIMIC experiment runner
│   └── mimic_pretrain.yaml             # MIMIC configuration
├── example.py           # Complete working example (Quick Start)
├── environment.yaml     # Conda environment
├── requirements.txt     # Dependencies
├── setup.py            # Package setup
├── LICENSE             # MIT License
└── README.md           # This file
```

---

## 📚 Supported Datasets

### Dataset Directory Setup

Create a `datasets/` directory in the project root and place all dataset files there with their respective filenames as shown below:

```bash
mkdir datasets
# Download datasets and place them in the datasets/ directory
```

### Dataset Table

| Dataset Name | Filename | URL |
|-------------|----------|-----|
| `airbnb` | `cleansed_listings_dec18.csv` | https://www.kaggle.com/datasets/tylerx/melbourne-airbnb-open-dataairbnb-listings-in-major-us-cities-deloitte-ml |
| `kick` | `kickstarter_train.csv` | https://www.kaggle.com/datasets/codename007/funding-successful-projects?select=train.csv |
| `cloth` | `Womens Clothing E-Commerce Reviews.csv` | https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews |
| `wine_10` / `wine_100` | `winemag-data-130k-v2.csv` | https://www.kaggle.com/datasets/zynicide/wine-reviews |
| `income` | `adult.csv` | https://www.kaggle.com/datasets/uciml/adult-census-income |
| `pet` | `petfinder_train.csv` | https://www.kaggle.com/competitions/petfinder-adoption-prediction/data |
| `jigsaw` | `jigsaw_train_100k.csv` | https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification |
| `mimic` | Special - See below | https://physionet.org/content/mimiciv/3.1/, https://physionet.org/content/mimic-iv-note/2.2/ |

### MIMIC Dataset

The MIMIC-IV dataset requires special handling. The configuration and MIMIC utilities (`load_mimic.py`, `mimic_utils.py`) are adapted from https://github.com/google-research/lanistr.

1. **Access**: Request access to MIMIC-IV at https://physionet.org/content/mimiciv/3.1/, https://physionet.org/content/mimic-iv-note/2.2/
2. **Download**: Download the MIMIC-IV dataset following PhysioNet instructions
3. **Preprocessing**: Follow the MedFuse extraction pipeline at https://github.com/nyuad-cai/MedFuse/tree/main/mimic4extract
4. **Configuration**: Update the paths in `src/mimic_pretrain.yaml` to point to where you extracted the MIMIC-IV and MIMIC-IV-Note data (follow the same structure as used in the Lanistr repository). Replace `YOUR_PATH` placeholders with your actual data directories:
   - `task_data_dir`: Path to extracted in-hospital-mortality data
   - `unimodal_data_dir`: Path to unimodal data directory
   - `preprocessed_data_dir`: Path to preprocessed data
   - `normalizer_file`: Path to normalizer.csv
   - `discretizer_config_path`: Path to discretizer_config.json
5. **Run**: Use `python src/main_mimic.py --version exp1` for MIMIC experiments

### Adding Custom Datasets

You can add any custom dataset by following these steps:

1. **Add your dataset file** to the `datasets/` directory
2. **Define dataset settings** in `src/tabulartextmultimodalfusion/settings.py` by adding a new configuration block
3. **Implement data loading** in `src/tabulartextmultimodalfusion/dataset.py` following existing preprocessing patterns
4. **Update your experiments** to include the new dataset name

---

## 🔍 Results and Analysis

### Performance Metrics
- **Accuracy**: Overall classification performance
- **F1-Score**: Balanced precision and recall
- **AUC-ROC**: Area under the ROC curve (binary classification)
- **Training Time**: Computational efficiency

### Expected Findings
- Cross-attention models typically outperform simple fusion baselines
- Numerical encoders provide significant improvements for datasets with complex numerical relationships
- Contrastive losses help when text and tabular modalities have different distributions

---

## 🙏 Attribution

Parts of the preprocessing pipeline (`settings.py`, `dataset.py`) are adapted from:

> Yury Petyushin, [Tabular Text Transformer](https://github.com/yury-petyushin/TabularTextTransformer), MIT License

We thank the original authors for their valuable contribution. This project modifies and builds upon that work with new architectures and optimization strategies.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

- **🐛 Bug Reports**: Open an issue with detailed reproduction steps
- **💡 Feature Requests**: Suggest new fusion strategies or loss functions
- **📊 New Datasets**: Add support for additional multimodal datasets
- **🔧 Code Improvements**: Submit pull requests for optimizations

### Development Guidelines
1. Follow existing code style and documentation patterns
2. Add tests for new model architectures
3. Update documentation for new features
4. Ensure reproducibility with fixed random seeds

---

## 🗺️ Roadmap

- [ ] **Transformer-based Fusion**: Implement transformer layers for cross-modal attention
- [ ] **Multi-task Learning**: Support for multiple prediction tasks
- [ ] **Hyperparameter Optimization**: Automated hyperparameter tuning
- [ ] **Model Interpretability**: Attention visualization and feature importance
- [ ] **Distributed Training**: Multi-GPU support for large-scale experiments
- [ ] **Pre-trained Models**: Release pre-trained checkpoints for common datasets

---