import os

# import libraries
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AdamW, DistilBertTokenizerFast, DistilBertModel, BertTokenizer
import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

# useful .py
from tabulartextmultimodalfusion.settings import * # settings
from tabulartextmultimodalfusion.dataset import * # data pre-processing
from tabulartextmultimodalfusion.models import * # models
from tabulartextmultimodalfusion.optimization import * # model selection, training, evaluation

import warnings
warnings.simplefilter('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

# VERSION = "exp1"
triplet_loader = None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    # select DATASET
    #DATASET choose in {"airbnb", "cloth", "jigsaw", "kick", "pet", "salary, "wine_10", "wine_100"}
    # DATASETS = ["airbnb", "kick", "cloth", "wine_10", "wine_100", "income", "pet", "jigsaw"]

    # Get from commend line all datasets as a list
    DATASETS = args.datasets

    # Model Architecture Selection Guide
    # =====================================

    # Base Model Architectures
    # -------------------------
    # Available base models for multimodal learning with text and tabular data:
    valid_base_models = [
        # Cross-modal attention models (our proposed approaches)
        'CrossAttention',           # Cross-attention between text and tabular features
        'CrossAttentionSkipNet',    # Cross-attention with skip connections
        
        # Fusion-based models
        'FusionSkipNet',           # Skip connections with feature fusion
        'CombinedModelGAT',        # Graph Attention Network for combined features
        
        # BERT-based approaches
        'LateFuseBERT',            # Late fusion of BERT text and tabular features
        'AllTextBERT',             # Convert tabular data to text for BERT processing
        'TabularForBert',          # Tabular data processing for BERT compatibility
        'BertWithTabular',         # BERT with additional tabular feature layers
        
        # Single-modality baselines
        'OnlyTabular',             # Tabular data only (baseline)
        'OnlyText'                 # Text data only (baseline)
    ]

    # Cross-Attention Configuration Options
    # ====================================

    # Fusion Methods for CrossAttention Models
    # ----------------------------------------
    # Format: {method}{dimensions}[s]
    # - method: Concat (concatenation) or SumW (weighted sum)
    # - dimensions: 2 or 4 (feature dimension multiplier)
    # - s suffix: applies BERT self-attention on final token embeddings (vs pre-trained embeddings)

    fusion_methods = {
        # Without BERT self-attention on final embeddings
        'pre_trained_embeddings': ['Concat2', 'Concat4', 'SumW2', 'SumW4'],
        
        # With BERT self-attention on final embeddings  
        'final_embeddings': ['Concat2s', 'Concat4s', 'SumW2s', 'SumW4s']
    }

    # Numerical Feature Encoders
    # ---------------------------
    # Transform numerical tabular features for better cross-modal alignment
    numerical_transforms = [
        'Fourier',      # Fourier feature encoding
        'FourierVec',   # Vectorized Fourier encoding
        'PosEnVec',     # Positional encoding vectors
        'RBF',          # Radial Basis Function encoding
        'RBFVec',       # Vectorized RBF encoding
        'Sigmoid',      # Sigmoid transformation
        'Chebyshev'     # Chebyshev polynomial encoding
    ]

    # Loss Functions for Cross-Modal Learning
    # ---------------------------------------
    loss_functions = [
        'MMD',          # Maximum Mean Discrepancy
        'MINE',         # Mutual Information Neural Estimation
        'InfoNCE',      # Info Noise Contrastive Estimation
        'Contrastive'   # Contrastive learning loss
    ]

    # Fusion Methods for CombinedModel Architecture
    # ---------------------------------------------
    combined_model_methods = ['Concat2', 'SumW2']  # Subset available for CombinedModel

    # Experimental Configurations
    # ===========================
    # Three main experiments comparing different aspects of multimodal fusion

    if args.version == "exp1":
        # Experiment 1: Architecture Comparison
        # Compare fusion architectures and baseline methods
        MODELS = [
            # Our proposed cross-attention variants
            "CrossAttentionSumW4", "CrossAttentionConcat4", 
            "CrossAttentionConcat4s", "CrossAttentionSumW4s",
            
            # Alternative fusion approaches
            "FusionSkipNet", "CombinedModelGAT",
            
            # BERT-based methods
            "BertWithTabular", "LateFuseBERT", "AllTextBERT",
            
            # Combined model variants
            "CombinedModelSumW2", "CombinedModelConcat2",
            
            # Single-modality baselines
            "OnlyTabular", "OnlyText"
        ]

    elif args.version == "exp2":
        # Experiment 2: Numerical Encoder Comparison
        # Test different numerical transformations with best-performing architecture (CrossAttentionConcat4)
        MODELS = [
            "CrossAttentionConcat4Fourier",
            "CrossAttentionConcat4RBF", 
            "CrossAttentionConcat4FourierVec",
            "CrossAttentionConcat4PosEnVec", 
            "CrossAttentionConcat4Chebyshev", 
            "CrossAttentionConcat4Sigmoid"
        ]

    elif args.version == "exp3":
        # Experiment 3: Loss Function Comparison  
        # Test different loss functions with best-performing architecture (CrossAttentionConcat4)
        MODELS = [
            "CrossAttentionConcat4MMD",
            "CrossAttentionConcat4MINE",
            "CrossAttentionConcat4InfoNCE",
            "CrossAttentionConcat4Contrastive"
        ]

    # Model Naming Convention
    # ======================
    # Models follow the pattern: {BaseModel}{FusionMethod}[{NumericalEncoder}][{LossFunction}]
    # 
    # Examples:
    # - CrossAttentionConcat4: Cross-attention with 4D concatenation fusion
    # - CrossAttentionConcat4s: Same as above but with self-attention on final embeddings  
    # - CrossAttentionConcat4Fourier: Cross-attention + Concat4 + Fourier encoding
    # - CrossAttentionConcat4MMD: Cross-attention + Concat4 + MMD loss

    for DATASET in DATASETS:
        FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT= load_settings(dataset = DATASET)
        # performance records
        perf_results = pd.DataFrame()
        i = 0

        for SEED in range(N_SEED):
            for MODEL_TYPE in MODELS:
                    start = time.time()
                    print(MODEL_TYPE)
                    # GPU or CPU
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
                    # load and prepare dataset
                    df = preprocess_dataset(DATASET, MODEL_TYPE)
        
                    # control randomness
                    random.seed(SEED)
                    np.random.seed(SEED)
                    torch.manual_seed(SEED)
                    torch.cuda.manual_seed(SEED)
        
                    # temporary dataframes to compute uncertainty metrics
                    uncertainty_results = pd.DataFrame()
                    val_uncertainty_results = pd.DataFrame()
        
                    perf_results.loc[i,"model type"] = MODEL_TYPE
                    perf_results.loc[i,"seed"] = SEED
        
                    # Train/Test split
                    df, target = train_test_split(df, test_size = split_val, random_state = SEED)
        
                    # text cleaning (keep only words and numbers)
                    df['clean_text'] = df[text_var].apply(lambda row:clean_text(row))
                    target['clean_text'] = target[text_var].apply(lambda row:clean_text(row))
        
                    # Load the specific tokenizer
                    if MODEL_TYPE != "CombinedModel6" or MODEL_TYPE != "CombinedModel2":
                        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)
                    else:
                        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
                    # text max length
                    MAX_LEN = int(np.quantile(df.apply(lambda row : len(tokenizer(row['clean_text']).input_ids), axis=1).values, q = [MAX_LEN_QUANTILE]).item())
                    MAX_LEN = min(MAX_LEN, 512) # maximum sequence length is 512 for DistilBERT
                    perf_results.loc[i,"max text length"] = MAX_LEN
        
                    # Numerical variables pre-processing
                    numerical_var_scaled = standardScaling(df, target, numerical_var)
                    NUM_NUMERICAL_VAR = len(numerical_var)
        
                    # Categorical variables pre-processing
                    categorical_var_oe, CAT_VOCAB_SIZES = ordinalEncoding(df, target, categorical_var)
                    NUM_CAT_VAR = len(categorical_var)
        
                    # train / validation split
                    df_train, df_validation = train_test_split(df, test_size = split_val, random_state = SEED)
                    perf_results.loc[i,"training size"] = df_train.shape[0]
                    perf_results.loc[i,"test size"] = target.shape[0]
        
                    # hyper-parameters
                    LR, BATCH_SIZE, D_FC, N_EPOCHS, N_HEADS, N_LAYERS = load_pretrained_settings()
                    perf_results.loc[i,"LR"] = LR
                    perf_results.loc[i,"BATCH_SIZE"] = BATCH_SIZE
                    perf_results.loc[i,"N_HEADS"] = N_HEADS
                    perf_results.loc[i,"N_LAYERS"] = N_LAYERS
        
                    # prepare the Tensor Datasets, including tokenization
                    if MODEL_TYPE == "CombinedModelGAT":
                        dataset_full = prepareTensorDatasetWithTokenizer(df, "clean_text", categorical_var_oe, numerical_var_scaled, 'Y', tokenizer, MAX_LEN, special_tokens=True, model_type = MODEL_TYPE)
                    else:
                        dataset_train = prepareTensorDatasetWithTokenizer(df_train, "clean_text", categorical_var_oe, numerical_var_scaled, 'Y', tokenizer, MAX_LEN, special_tokens=True, model_type = MODEL_TYPE)
                        dataset_validation = prepareTensorDatasetWithTokenizer(df_validation, "clean_text", categorical_var_oe, numerical_var_scaled, 'Y', tokenizer, MAX_LEN, special_tokens=True, model_type = MODEL_TYPE)
                        dataset_target = prepareTensorDatasetWithTokenizer(target, "clean_text", categorical_var_oe, numerical_var_scaled, 'Y', tokenizer, MAX_LEN, special_tokens=True, model_type = MODEL_TYPE)
            
                        # data loaders
                        loader_train = DataLoader(dataset_train, sampler = RandomSampler(dataset_train), batch_size = BATCH_SIZE)
                        loader_validation = DataLoader(dataset_validation, sampler = SequentialSampler(dataset_validation),batch_size = BATCH_SIZE)
                        loader_target = DataLoader(dataset_target, sampler = SequentialSampler(dataset_target),batch_size = BATCH_SIZE)
                        
                        if "Contrastive" in MODEL_TYPE:
                            triplet_dataset = TripletDataset(loader_train)
                            triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)
                        
        
                    # Load Bert with a linear classification layer
                    if "LANISTRR" == MODEL_TYPE:
                        bert_config = transformers.BertConfig()
                        bert_config.is_decoder = False
                        BERT_model = transformers.BertModel(
                            bert_config, add_pooling_layer=False
                        ).from_pretrained("bert-base-uncased").to(device)
                    else:
                        BERT_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
                        
                        
                    if MODEL_TYPE == "CombinedModelGAT":
                        input_ids = dataset_full.tensors[0]      # Tokenized text input (BERT input)
                        categoricals = dataset_full.tensors[1]    # Categorical features
                        numericals = dataset_full.tensors[2]      # Numerical features
                        labels = dataset_full.tensors[3]          # Target labels
                        attention_masks = dataset_full.tensors[4] # Attention masks for BERT
                        
                        encode_all, encode_text, encode_tabular, encode_numeric, encode_cat = encode_text_numeric(input_ids, attention_masks, categoricals, numericals, BERT_model, tokenizer, CAT_VOCAB_SIZES, device)
                        edge_index_text, edge_weights1 = build_threshold_knn_graph(encode_text, k=5, threshold_factor=1)
                        edge_index_numeric, edge_weights2 = build_threshold_knn_graph(encode_numeric, k=5, threshold_factor=1)
                        edge_index_cat = build_knn_graph(encode_cat, k=5, metric='hamming')
                        
                        # Combine the edges into a single tensor
                        edge_index = torch.cat([edge_index_text, edge_index_numeric], dim=1)

                        edge_weights = torch.cat([edge_weights1.unsqueeze(0), edge_weights2.unsqueeze(0)], dim=1).squeeze(0)

                        graph_data = Data(input_ids=input_ids, attention_masks=attention_masks, categoricals=categoricals, numericals=numericals, edge_index=edge_index, edge_attr=edge_weights, y=labels)
                        
                        train_mask, val_mask, test_mask = split_data(len(encode_all), SEED)

                        # Add masks to the graph data
                        graph_data.train_mask = train_mask
                        graph_data.val_mask = val_mask
                        graph_data.test_mask = test_mask

                        train_loader = NeighborLoader(
                            graph_data,
                            num_neighbors=[3, 3],  # Number of neighbors to sample at each layer
                            batch_size=32,  # Number of root nodes per batch
                            input_nodes=graph_data.train_mask,  # Sample only from training nodes
                            shuffle=True
                        )

                        val_loader = NeighborLoader(
                            graph_data,
                            num_neighbors=[3, 3],
                            batch_size=32,
                            input_nodes=graph_data.val_mask,
                            shuffle=False
                        )

                        test_loader = NeighborLoader(
                            graph_data,
                            num_neighbors=[3, 3],
                            batch_size=32,
                            input_nodes=graph_data.test_mask,
                            shuffle=False
                        )
                        
                        
                        
                    if MODEL_TYPE not in ["LateFuseBERT", "AllTextBERT", "OnlyTabular", "OnlyText"]:
                    #best right now
                        D_FC = 256 #best 256
                        D_FF = 256 #best 256
                        ca_dropout = 0.1#best0.1
                        N_HEADS = 8#best 8
                        DROPOUT = 0.1#prev: 0.2 best:0.1
                        LR = 0.0001
                        N_LAYERS = 4
                    # model initialization
                    torch.manual_seed(SEED)
                    model = init_model(model_type = MODEL_TYPE,
                                    d_model = BERT_model.embeddings.word_embeddings.embedding_dim, # dimension = 768 for BERT family
                                    max_len = "", # not used here
                                    vocab_size = "", # not used here
                                    cat_vocab_sizes = CAT_VOCAB_SIZES,
                                    num_cat_var = NUM_CAT_VAR,
                                    num_numerical_var = NUM_NUMERICAL_VAR,
                                    quantiles = "", # not used here
                                    n_heads = N_HEADS,
                                    d_ff = D_FF, # not used here
                                    n_layers = N_LAYERS,
                                    dropout = DROPOUT,
                                    d_fc = D_FC,
                                    n_classes = N_CLASSES,
                                    seed = SEED,
                                    device=device,
                                    text_model = BERT_model,
                                    ca_dropout=ca_dropout).to(device)
        
                    # number of trainable parameters
                    perf_results.loc[i,"trainable parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
                    # optimizer
                    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
                    # training
                    print("Start Training:")
                    
                    if MODEL_TYPE == "CombinedModelGAT":
                        training_mode="graph"
                    elif "Contrastive" in MODEL_TYPE:
                        training_mode="contrastive"
                    else:
                        training_mode="standard"

                    model, epochs = unified_training(model, DATASET, MODEL_TYPE, train_loader, N_EPOCHS, val_loader, CRITERION, optimizer, FACTOR, SEED, verbose=True, device=device, training_mode=training_mode, triplet_loader=triplet_loader)
                    perf_results.loc[i,"epochs"] = epochs
        
                    # model evaluation
                    model.eval()
                    target_perf = unified_performance(model, DATASET+"_test", loader_target, MODEL_TYPE, SEED, device, training_mode=training_mode, data_split="test")
                    accuracy = target_perf["accuracy"]
                    micro_f1 = target_perf["micro_f1"]
                    macro_f1 = target_perf["macro_f1"]
                    auc = target_perf["auc"]
                        
                    print(target_perf)
                    perf_results.loc[i,"accuracy (Target)"] = accuracy
                    perf_results.loc[i,"micro_f1 (Target)"] = micro_f1
                    perf_results.loc[i,"macro_f1 (Target)"] = macro_f1
                    perf_results.loc[i,"auc (Target)"] = auc
                    print("Test ACC:", target_perf)
        
                    elapsed_time = time.time()-start
                    perf_results.loc[i,"time"] = elapsed_time
        
                    i+=1
            
        directory = 'Outputs'+ "_" + args.version
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        f1 = f'output_{DATASET}.csv'
        f1 = os.path.join(directory, f1)
        perf_results.to_csv(f1, index=False)


if __name__ == "__main__":
    main()