import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
import pandas as pd
import numpy as np
import time
import random
import warnings
import argparse

from tabulartextmultimodalfusion.settings import *
from tabulartextmultimodalfusion.models import *
from tabulartextmultimodalfusion.optimization import *
from tabulartextmultimodalfusion.load_mimic import load_mimic

# Add GAT-specific imports
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)
from omegaconf import OmegaConf

def get_mimic_args():
    """Load MIMIC configuration from mimic_pretrain.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src/mimic_pretrain.yaml')
    config = OmegaConf.load(config_path)
    # Override task to finetune since we only support finetune
    config.task = 'finetune'
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    # Model selection based on version
    if args.version == "exp1":
        # Experiment 1: Architecture Comparison
        MODELS = [
            "CrossAttentionConcat4", "CrossAttentionSumW4",
            "CrossAttentionConcat4s", "CrossAttentionSumW4s",
            "FusionSkipNet", "CombinedModelGAT",
            "BertWithTabular", 
            # "AllTextBERT",
            "CombinedModelSumW2", "CombinedModelConcat2",
            "OnlyTabular", "OnlyText"
        ]
    elif args.version == "exp2":
        # Experiment 2: Numerical Encoder Comparison
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
        MODELS = [
            "CrossAttentionConcat4MMD",
            "CrossAttentionConcat4MINE",
            "CrossAttentionConcat4InfoNCE",
            "CrossAttentionConcat4Contrastive"
        ]

    DATASET = "mimic"
    perf_results = pd.DataFrame()
    i = 0
    #N_SEED = 1  # You can set this to your desired number of seeds

    # Load settings for mimic
    FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT = load_settings(dataset=DATASET)
    
    LR, BATCH_SIZE, D_FC, N_EPOCHS, N_HEADS, N_LAYERS = load_pretrained_settings()

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    # Load MIMIC dataset using the provided pipeline
    from omegaconf import OmegaConf
    # Assume settings.py provides a function to get omegaconf config for mimic
    mimic_args = get_mimic_args()  # You may need to implement this in settings.py
    dataset = load_mimic(mimic_args, tokenizer)
    # Remove old split logic from here
    # train_set = dataset['train']
    # val_set = dataset['valid']
    # test_set = dataset['test']

    for SEED in range(N_SEED):
        print("SEED:", SEED)
        # Concatenate all splits into one
        train_set = dataset['train']
        val_set = dataset['valid']
        test_set = dataset['test']
        

        # Print actual split sizes
        print(f"Actual split sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

        # DataLoaders
        BATCH_SIZE = 32  # Or set from settings

        for MODEL_TYPE in MODELS:
            start = time.time()
            print(MODEL_TYPE)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)

            perf_results.loc[i, "model type"] = MODEL_TYPE
            perf_results.loc[i, "seed"] = SEED
            perf_results.loc[i, "training size"] = len(train_set)
            perf_results.loc[i, "test size"] = len(test_set)

            # Load Bert backbone
            BERT_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
            
            if MODEL_TYPE not in ["LateFuseBERT", "AllTextBERT", "OnlyTabular", "OnlyText"]:
                D_FC = 256 #best 256
                D_FF = 256 #best 256
                ca_dropout = 0.1#best0.1
                N_HEADS = 8#best 8
                DROPOUT = 0.1#prev: 0.2 best:0.1
                LR = 0.0001
                N_LAYERS = 4

            # Special handling for GAT model
            if MODEL_TYPE == "CombinedModelGAT":
                # Create a combined dataset for GAT preprocessing
                # Combine all splits for graph construction
                all_input_ids = torch.cat([train_set.tensors[0], val_set.tensors[0], test_set.tensors[0]], dim=0)
                all_categoricals = torch.cat([train_set.tensors[1], val_set.tensors[1], test_set.tensors[1]], dim=0)
                all_numericals = torch.cat([train_set.tensors[2], val_set.tensors[2], test_set.tensors[2]], dim=0)
                all_labels = torch.cat([train_set.tensors[3], val_set.tensors[3], test_set.tensors[3]], dim=0)
                all_attention_masks = torch.cat([train_set.tensors[4], val_set.tensors[4], test_set.tensors[4]], dim=0)
                
                # Get vocabulary sizes from dataset
                CAT_VOCAB_SIZES = dataset['cat_vocab_sizes']
                
                print("Building graph for GAT model...")
                
                # Encode all data for graph construction
                encode_all, encode_text, encode_tabular, encode_numeric, encode_cat = encode_text_numeric(
                    all_input_ids, all_attention_masks, all_categoricals, all_numericals, 
                    BERT_model, tokenizer, CAT_VOCAB_SIZES, device
                )
                
                # Build graphs
                edge_index_text, edge_weights1 = build_threshold_knn_graph(encode_text, k=5, threshold_factor=1)
                edge_index_numeric, edge_weights2 = build_threshold_knn_graph(encode_numeric, k=5, threshold_factor=1)
                if encode_cat.shape[1] > 0:
                    edge_index_cat = build_knn_graph(encode_cat, k=5, metric='hamming')
                else:
                    edge_index_cat = None  # No categorical features present
                
                # Combine the edges into a single tensor
                edge_index = torch.cat([edge_index_text, edge_index_numeric], dim=1)
                edge_weights = torch.cat([edge_weights1.unsqueeze(0), edge_weights2.unsqueeze(0)], dim=1).squeeze(0)

                # Create graph data
                graph_data = Data(
                    input_ids=all_input_ids, 
                    attention_masks=all_attention_masks, 
                    categoricals=all_categoricals, 
                    numericals=all_numericals, 
                    edge_index=edge_index, 
                    edge_attr=edge_weights, 
                    y=all_labels
                )
                
                # Create train/val/test masks based on original split sizes
                train_size = len(train_set)
                val_size = len(val_set)
                test_size = len(test_set)
                total_size = train_size + val_size + test_size
                
                train_mask = torch.zeros(total_size, dtype=torch.bool)
                val_mask = torch.zeros(total_size, dtype=torch.bool)
                test_mask = torch.zeros(total_size, dtype=torch.bool)
                
                train_mask[:train_size] = True
                val_mask[train_size:train_size + val_size] = True
                test_mask[train_size + val_size:] = True

                # Add masks to the graph data
                graph_data.train_mask = train_mask
                graph_data.val_mask = val_mask
                graph_data.test_mask = test_mask

                # Create NeighborLoaders for GAT
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
                
                training_mode = "graph"
                
            else:
                # Standard DataLoaders for non-GAT models
                train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
                test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
                training_mode = "standard"

            # Model initialization (adapt as needed for your models)
            torch.manual_seed(SEED)
            model = init_model(
                model_type=MODEL_TYPE,
                d_model=BERT_model.embeddings.word_embeddings.embedding_dim,
                max_len="",
                vocab_size="",
                cat_vocab_sizes=dataset['cat_vocab_sizes'],  # Set as needed
                num_cat_var=dataset['num_cat_var'],         # Set as needed
                num_numerical_var=dataset['num_numerical_var'],   # Set as needed
                quantiles="",
                n_heads=N_HEADS,
                d_ff=D_FF,
                n_layers=N_LAYERS,
                dropout=DROPOUT,
                d_fc=D_FC,
                n_classes=N_CLASSES,
                seed=SEED,
                device=device,
                text_model=BERT_model,
                ca_dropout=ca_dropout
            ).to(device)

            perf_results.loc[i, "trainable parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            print("Start Training:")
            model, epochs = fast_unified_training(
                model, DATASET, MODEL_TYPE, train_loader, N_EPOCHS, val_loader, 
                CRITERION, optimizer, FACTOR, SEED, verbose=True, device=device, 
                training_mode=training_mode, triplet_loader=None
            )
            perf_results.loc[i, "epochs"] = epochs

            model.eval()
            target_perf = fast_unified_performance(
                model, DATASET+"_test", test_loader, MODEL_TYPE, SEED, device, 
                training_mode=training_mode, data_split="test"
            )
            accuracy = target_perf["accuracy"]
            micro_f1 = target_perf["micro_f1"]
            macro_f1 = target_perf["macro_f1"]
            auc = target_perf["auc"]
            print(target_perf)
            perf_results.loc[i, "accuracy (Target)"] = accuracy
            perf_results.loc[i, "micro_f1 (Target)"] = micro_f1
            perf_results.loc[i, "macro_f1 (Target)"] = macro_f1
            perf_results.loc[i, "auc (Target)"] = auc
            print("Test ACC:", target_perf)

            elapsed_time = time.time() - start
            perf_results.loc[i, "time"] = elapsed_time
            i += 1

    directory = f'outputs_{args.version}_mimic'
    os.makedirs(directory, exist_ok=True)
    f1 = f'output_{DATASET}.csv'
    f1 = os.path.join(directory, f1)
    perf_results.to_csv(f1, index=False)

if __name__ == "__main__":
    main()