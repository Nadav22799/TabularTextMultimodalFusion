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
python main.py --version exp1

# Run experiment 2 (numerical encoders)
python main.py --version exp2

# Run experiment 3 (loss functions)
python main.py --version exp3
```

### Customization Options

#### 1. **Version Selection** (CLI)
Choose experiment type via command line:
```bash
python main.py --version exp1  # Architecture comparison
python main.py --version exp2  # Numerical encoder comparison  
python main.py --version exp3  # Loss function comparison
```

#### 2. **Dataset Selection** (Manual)
Edit `main.py` to customize datasets:
```python
DATASETS = ["wine_10", "airbnb", "kick"]  # Select from supported datasets
```

#### 3. **Model Selection** (Automatic by Version)
Models are automatically selected based on the version:
- **`exp1`**: Tests all architecture variants and baselines
- **`exp2`**: Tests numerical encoders with `CrossAttentionConcat4`
- **`exp3`**: Tests loss functions with `CrossAttentionConcat4`

Or manually override in `main.py`:
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

## 📦 Installation

This project uses a two-step installation process to ensure that all dependencies and the correct Python version are set up correctly. First, you create a dedicated environment using `conda`, and then you install the package into that environment using `pip`.

### Step 1: Create the Conda Environment

This command uses the `environment.yaml` file to create a new conda environment named `TTMF` with all the necessary base packages, including the correct Python version.

```bash
# Create the conda environment
conda env create -f environment.yaml

# Activate the newly created environment
conda activate TTMF
```

### Step 2: Install the Package (Editable Mode)

Once the environment is active, install the `TabularTextMultimodalFusion` package in "editable" mode. This is the recommended approach for development, as any changes you make to the source code will be immediately available without needing to reinstall.

The `setup.py` file manages this process, using `requirements.txt` for additional `pip`-based dependencies.

```bash
# Install the package in editable mode
pip install -e .
```

Now your environment is fully set up and ready for running experiments.

### Package Structure (for pip installation)

```
TabularTextMultimodalFusion/
├── TabularTextMultimodalFusion/    # Main package
│   ├── __init__.py
│   ├── models.py         # Model architectures
│   ├── dataset.py        # Data loading and preprocessing
│   ├── settings.py       # Configuration
│   └── utils.py          # Utilities
├── main.py               # Experiment runner
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

---

## 📚 Supported Datasets

| Dataset | Domain | Text Feature | Tabular Features | Classes |
|---------|--------|--------------|------------------|---------|
| `airbnb` | Real Estate | Property descriptions | Price, location, amenities | 2 |
| `kick` | E-commerce | Product descriptions | Seller metrics, pricing | 2 |
| `cloth` | Fashion | Product titles | Brand, category, ratings | Multiple |
| `wine_10` | Food & Beverage | Wine descriptions | Chemical composition | 10 |
| `wine_100` | Food & Beverage | Wine descriptions | Chemical composition | 100 |
| `income` | Demographics | Job descriptions | Personal attributes | 2 |
| `pet` | Animals | Pet descriptions | Breed, age, characteristics | Multiple |
| `jigsaw` | Social Media | Comments/posts | User metadata | 2 |

### Adding Custom Datasets

1. **Define dataset settings** in `settings.py`
2. **Implement data loading** in `dataset.py`
3. **Add preprocessing logic** following existing patterns

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

## 📖 Citation

If you find this repository useful, please consider citing it:

```bibtex
@misc{tabulartextmultimodalfusion2025,
  author = {Nadav Cohen},
  title = {TabularTextMultimodalFusion: Unified Architectures for Text + Tabular Data Fusion},
  year = {2025},
  howpublished = {\url{https://github.com/your-username/TabularTextMultimodalFusion}},
  note = {Work inspired by Tabular Text Transformer}
}
```

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