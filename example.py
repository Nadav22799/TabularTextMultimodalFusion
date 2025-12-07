"""
Simple example demonstrating how to use TabularTextMultimodalFusion package.

IMPORTANT: Before running this example, make sure you have installed the package:
    pip install -e .

This example shows how to:
1. Create synthetic data
2. Initialize a CrossAttention model with Concat4 fusion
3. Make predictions
"""

import torch
import pandas as pd
from transformers import DistilBertTokenizer

from tabulartextmultimodalfusion.models import (
    CrossAttention,
    CombinedModelConcat4,
    TabularEmbedding
)
from tabulartextmultimodalfusion.settings import load_settings

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 1. Create Synthetic Data ====================
print("\n1. Creating synthetic data...")

# Create a small synthetic dataset
n_samples = 100
data = {
    'text': [f"This is sample text number {i} for classification." for i in range(n_samples)],
    'category_1': [f'cat_{i%5}' for i in range(n_samples)],  # 5 categories
    'category_2': [f'type_{i%3}' for i in range(n_samples)], # 3 categories
    'numeric_1': torch.randn(n_samples).tolist(),
    'numeric_2': torch.randn(n_samples).abs().tolist(),
    'label': torch.randint(0, 2, (n_samples,)).tolist()  # Binary classification
}
df = pd.DataFrame(data)

print(f"Created dataset with {len(df)} samples")
print(f"Categorical variables: category_1, category_2")
print(f"Numerical variables: numeric_1, numeric_2")

# ==================== 2. Prepare Data ====================
print("\n2. Preparing data...")

# Tokenize text
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 128

# Tokenize all texts
tokenized = tokenizer(
    df['text'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=max_len,
    return_tensors='pt'
)

input_ids = tokenized['input_ids'].to(device)
attention_mask = tokenized['attention_mask'].to(device)

# Prepare categorical data (encode as integers)
cat1_vocab = {v: i for i, v in enumerate(df['category_1'].unique())}
cat2_vocab = {v: i for i, v in enumerate(df['category_2'].unique())}

cat_data = torch.tensor([
    [cat1_vocab[c1], cat2_vocab[c2]]
    for c1, c2 in zip(df['category_1'], df['category_2'])
]).to(device)

# Prepare numerical data
num_data = torch.tensor(
    df[['numeric_1', 'numeric_2']].values,
    dtype=torch.float32
).to(device)

# Labels
labels = torch.tensor(df['label'].tolist()).to(device)

print(f"Input IDs shape: {input_ids.shape}")
print(f"Categorical data shape: {cat_data.shape}")
print(f"Numerical data shape: {num_data.shape}")

# ==================== 3. Initialize Model ====================
print("\n3. Initializing CrossAttention + Concat4 model...")

# Model hyperparameters
d_model = 768  # DistilBERT embedding dimension
num_classes = 2  # Binary classification
num_categorical = 2
num_numerical = 2
cat_embed_dim = 768  # Embedding dimension for categorical variables must match d_model
cat_vocab_sizes = [len(cat1_vocab), len(cat2_vocab)]

# Create tabular embedding layer
tab_embed = TabularEmbedding(
    cat_vocab_sizes=cat_vocab_sizes,
    num_numerical_vars=num_numerical,
    embed_dim=cat_embed_dim,
).to(device)

# Create CrossAttention model
cross_attention = CrossAttention(
    TabEmbed=tab_embed,
    num_numerical_var=num_numerical,
    nhead=8,
    n_layers=2,
    cat_embed_dims=cat_embed_dim,
    num_classes=num_classes,
    dropout=0.1,
    d_model=d_model,
    device=device,
    bert_self_attention=True,  # Use 's' suffix models
    d_ff=768,
    d_fc=128,
    ca_dropout=0.1
).to(device)

# Wrap with CombinedModelConcat4 for Concat4 fusion
model = CombinedModelConcat4(
    MultiModelsObj=CrossAttention,
    TabEmbed=tab_embed,
    num_cat_var=num_categorical,
    num_numerical_var=num_numerical,
    nhead=8,
    n_layers=2,
    cat_embed_dims=cat_embed_dim,
    num_classes=num_classes,
    dropout=0.1,
    d_model=d_model,
    device=device,
    bert_self_attention=True,
    d_ff=768,
    d_fc=128,
    ca_dropout=0.1
).to(device)

print(f"Model initialized: CombinedModelConcat4 with CrossAttention")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 4. Make Predictions ====================
print("\n4. Making predictions (inference mode)...")

model.eval()
with torch.no_grad():
    # Take a small batch for demo
    batch_size = 8
    batch_input_ids = input_ids[:batch_size]
    batch_attention_mask = attention_mask[:batch_size]
    batch_cat = cat_data[:batch_size]
    batch_num = num_data[:batch_size]

    # Forward pass
    logits, tab_emb, text_emb = model(
        batch_input_ids,
        batch_attention_mask,
        batch_cat,
        batch_num
    )

    # Get predictions
    predictions = torch.argmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)

    print(f"\nPredictions for first {batch_size} samples:")
    for i in range(batch_size):
        print(f"  Sample {i}: Class {predictions[i].item()} "
              f"(prob: {probabilities[i][predictions[i]].item():.3f})")

# ==================== 5. Training Example (Optional) ====================
print("\n5. Training example (one step)...")

# Set model to training mode
model.train()

# Simple training step
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Single training batch
logits, tab_emb, text_emb = model(
    batch_input_ids,
    batch_attention_mask,
    batch_cat,
    batch_num
)

loss = criterion(logits, labels[:batch_size])
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")

print("\n✅ Example completed successfully!")
print("\nNext steps:")
print("- Load your own dataset")
print("- Adjust model hyperparameters")
print("- Implement full training loop")
print("- See GitHub repo for more examples: https://github.com/nadav22799/TabularTextMultimodalFusion")
