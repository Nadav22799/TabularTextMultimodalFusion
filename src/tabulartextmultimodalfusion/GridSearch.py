import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer
import optuna
from optuna.trial import TrialState
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import random
import json
import copy
from collections import defaultdict

# Import your custom modules
from settings import *
from dataset import *
from models import *
from optimization import *

# Configuration
VERSION = "Unified_Experiment_v1"
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets for hyperparameter tuning
DATASETS = ["airbnb", "kick", "wine_10"]

# Models for different experiment types
GENERAL_MODELS = ["CombinedModelSumW2", "CombinedModelConcat2", "CrossAttentionConcat4", 
                  "CrossAttentionSumW4", "FusionSkipNet", "BertWithTabular",
                  "CrossAttentionConcat4s", "CrossAttentionSumW4s"]

CA_MODELS = ["CrossAttentionConcat4", "CrossAttentionSumW4", 
             "CrossAttentionConcat4s", "CrossAttentionSumW4s"]

LOSS_MODELS = ["CrossAttentionConcat4InfoNCE", "CrossAttentionConcat4MINE", 
               "CrossAttentionConcat4Contrastive", "CrossAttentionConcat4MMD"]

def run_unified_hyperparameter_optimization(experiment_type="general"):
    """
    Run unified hyperparameter optimization for different experiment types.
    
    Args:
        experiment_type: "general" for general hyperparameter tuning, 
                        "CA" for CrossAttention-specific tuning,
                        "loss" for loss-specific tuning
    """
    
    # Validate experiment type
    valid_types = ["general", "CA", "loss"]
    if experiment_type not in valid_types:
        raise ValueError(f"experiment_type must be one of {valid_types}")
    
    # Select models based on experiment type
    if experiment_type == "general":
        MODELS = GENERAL_MODELS
    elif experiment_type == "CA":
        MODELS = CA_MODELS
    else:  # loss
        MODELS = LOSS_MODELS
    
    # Initialize storage
    all_trials = []
    all_datasets = {}
    
    # Prepare datasets
    print(f"\n{'='*60}")
    print(f"Running {experiment_type} hyperparameter optimization")
    print(f"{'='*60}\n")
    
    for dataset_name in DATASETS:
        FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, \
        WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT = load_settings(dataset=dataset_name)
        
        print(f"\n=== Processing Dataset: {dataset_name} ===")
        all_datasets[dataset_name] = {}
        all_datasets[dataset_name]["WEIGHT_DECAY"] = WEIGHT_DECAY
        all_datasets[dataset_name]["N_CLASSES"] = N_CLASSES
        all_datasets[dataset_name]["CRITERION"] = CRITERION
        
        # Preprocessing
        df = preprocess_dataset(dataset_name, None)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        
        # Split Train/Test
        df, target = train_test_split(df, test_size=0.2, random_state=SEED)
        
        # Clean Text
        df['clean_text'] = df[text_var].apply(clean_text)
        target['clean_text'] = target[text_var].apply(clean_text)
        
        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Max Length
        MAX_LEN = int(np.quantile(
            df.apply(lambda row: len(tokenizer(row['clean_text']).input_ids), axis=1).values, 
            q=[0.95]
        ).item())
        MAX_LEN = min(MAX_LEN, 512)
        
        # Numerical & Categorical Encoding
        numerical_var_scaled = standardScaling(df, target, numerical_var)
        NUM_NUMERICAL_VAR = len(numerical_var)
        all_datasets[dataset_name]["numerical_var"] = NUM_NUMERICAL_VAR
        
        categorical_var_oe, CAT_VOCAB_SIZES = ordinalEncoding(df, target, categorical_var)
        all_datasets[dataset_name]["cat_vocab_sizes"] = copy.deepcopy(CAT_VOCAB_SIZES)
        NUM_CAT_VAR = len(categorical_var)
        all_datasets[dataset_name]["categorical_var"] = NUM_CAT_VAR
        
        # Train/Validation Split
        df_train, df_validation = train_test_split(df, test_size=0.2, random_state=SEED)
        
        # TensorDatasets
        dataset_train = prepareTensorDatasetWithTokenizer(
            df_train, "clean_text", categorical_var_oe, numerical_var_scaled, 
            'Y', tokenizer, MAX_LEN, special_tokens=True, model_type=None
        )
        dataset_validation = prepareTensorDatasetWithTokenizer(
            df_validation, "clean_text", categorical_var_oe, numerical_var_scaled, 
            'Y', tokenizer, MAX_LEN, special_tokens=True, model_type=None
        )
        
        # Criterion
        criterion = torch.nn.CrossEntropyLoss()
        all_datasets[dataset_name]["train"] = copy.deepcopy(dataset_train)
        all_datasets[dataset_name]["val"] = copy.deepcopy(dataset_validation)
    
    # Run optimization for each model
    for model_type in MODELS:
        print(f"\n--- Model: {model_type} ---")
        
        if experiment_type in ["general", "CA"]:
            trials = hp_optimization_pretrained(
                model_type=model_type,
                criterion=criterion,
                seed=SEED,
                device=device,
                datasets_names=DATASETS,
                all_datasets=all_datasets,
                experiment_type=experiment_type  # Pass experiment type instead of exp1
            )
        else:  # loss experiment
            trials = hp_optimization_losses(
                model_type=model_type,
                criterion=criterion,
                seed=SEED,
                device=device,
                datasets_names=DATASETS,
                all_datasets=all_datasets
            )
        
        for trial in trials:
            params_key = "_".join(f"{key}={value}" for key, value in sorted(trial.params.items()))
            all_trials.append({
                "model_type": model_type,
                "params_key": params_key,
                "params": trial.params,
                "value": trial.value,
            })
    
    # Process results based on experiment type
    if experiment_type in ["general", "CA"]:
        # Find best params across all datasets
        params_sums = defaultdict(float)
        for trial in all_trials:
            params_sums[trial['params_key']] += trial['value']
        
        best_params_key = max(params_sums, key=lambda k: params_sums[k])
        best_params_sum = params_sums[best_params_key]
        best_params = next(trial['params'] for trial in all_trials if trial['params_key'] == best_params_key)
        
        print("\n=== Best Hyperparameters Across All Datasets ===")
        print(f"Best summed value: {best_params_sum}")
        print(f"Best hyperparameters:")
        print(best_params)
        
        # Save results
        output_filename = f"best_params_overall_{experiment_type.lower()}"
        pd.DataFrame([{
            "params": best_params,
            "summed_value": best_params_sum,
        }]).to_csv(f"{output_filename}.csv", index=False)
        
        with open(f"{output_filename}.json", "w") as f:
            json.dump({
                "params": best_params,
                "summed_value": best_params_sum,
                "experiment_type": experiment_type
            }, f, indent=4)
    
    else:  # loss experiment
        # Find best params per model type
        model_param_sums = defaultdict(float)
        param_lookup = {}
        
        for trial in all_trials:
            key = (trial['model_type'], trial['params_key'])
            model_param_sums[key] += trial['value']
            param_lookup[key] = trial['params']
        
        best_params_per_model = {}
        best_scores_per_model = {}
        
        for (model_type, params_key), total_value in model_param_sums.items():
            if model_type not in best_scores_per_model or total_value > best_scores_per_model[model_type]:
                best_scores_per_model[model_type] = total_value
                best_params_per_model[model_type] = {
                    "summed_value": total_value,
                    "params": param_lookup[(model_type, params_key)]
                }
        
        # Save to JSON
        with open("best_params_per_model_loss.json", "w") as f:
            json.dump(best_params_per_model, f, indent=4)
        
        print("\nSaved best hyperparameters per model to best_params_per_model_loss.json")
        
        # Print results
        print("\n=== Best Hyperparameters Per Model ===")
        for model_type, info in best_params_per_model.items():
            print(f"\nModel: {model_type}")
            print(f"Best summed value: {info['summed_value']}")
            print(f"Best params: {info['params']}")
    
    return all_trials

def main():
    """
    Main function to run all experiments
    """
    # Run general hyperparameter optimization
    print("\n" + "="*80)
    print("EXPERIMENT 1: General Hyperparameter Optimization")
    print("="*80)
    general_trials = run_unified_hyperparameter_optimization(experiment_type="general")
    
    # Run CrossAttention-specific hyperparameter optimization
    print("\n" + "="*80)
    print("EXPERIMENT 2: CrossAttention-Specific Hyperparameter Optimization")
    print("="*80)
    ca_trials = run_unified_hyperparameter_optimization(experiment_type="CA")
    
    # Run loss-specific hyperparameter optimization
    print("\n" + "="*80)
    print("EXPERIMENT 3: Loss-Specific Hyperparameter Optimization")
    print("="*80)
    loss_trials = run_unified_hyperparameter_optimization(experiment_type="loss")
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)
    
    # Summary of output files
    print("\nOutput files generated:")
    print("- best_params_overall_general.json: Best parameters for general optimization")
    print("- best_params_overall_ca.json: Best parameters for CrossAttention optimization")
    print("- best_params_per_model_loss.json: Best parameters per model for loss optimization")

if __name__ == "__main__":
    # You can run specific experiments or all of them
    # Option 1: Run all experiments
    main()
    
    # Option 2: Run specific experiments
    # run_unified_hyperparameter_optimization(experiment_type="general")
    # run_unified_hyperparameter_optimization(experiment_type="CA")
    # run_unified_hyperparameter_optimization(experiment_type="loss")