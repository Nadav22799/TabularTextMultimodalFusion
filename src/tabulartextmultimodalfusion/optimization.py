from torch.optim.lr_scheduler import ExponentialLR
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from models import * # models
from settings import * # settings
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
import pickle
import os
from sklearn.metrics import f1_score, roc_auc_score
from dataset import * # data pre-processing

VERSION = "exp1"
L1 = False
L2 = False

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = F.softmax(anchor, dim=-1)
        
        # Convert anchor probabilities to log-probabilities
        anchor_log_prob = torch.log(anchor + 1e-8)  # Stability with epsilon

        # KL Divergence: Anchor (prob) vs Positive (logits)
        pos_dist = F.kl_div(anchor_log_prob, F.softmax(positive, dim=-1)+ 1e-8, reduction='batchmean')

        # KL Divergence: Anchor (prob) vs Negative (logits)
        neg_dist = F.kl_div(anchor_log_prob, F.softmax(negative, dim=-1)+ 1e-8, reduction='batchmean')

        # Triplet loss calculation
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
        
def info_nce_loss(text_repr, tab_repr, temperature=0.1):
    """
    text_repr: (batch_size, dim) – text embeddings
    tab_repr:  (batch_size, dim) – tabular embeddings
    """
    # Normalize
    text_norm = F.normalize(text_repr, dim=1)
    tab_norm = F.normalize(tab_repr, dim=1)

    # Similarity matrix
    logits = torch.matmul(text_norm, tab_norm.T)  # (B, B)
    
    # Ground truth labels (diagonal = positive pairs)
    labels = torch.arange(text_repr.size(0)).to(text_repr.device)

    # Scale by temperature
    logits = logits / temperature

    loss_t2v = F.cross_entropy(logits, labels)
    loss_v2t = F.cross_entropy(logits.T, labels)

    return (loss_t2v + loss_v2t) / 2
        
def gaussian_kernel(x, y, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between x and y.
    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    y_norm = y.pow(2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.T)
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_loss(x, y, sigma=1.0):
    """
    x: (batch_size, dim) – text representations
    y: (batch_size, dim) – tabular representations
    """
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)

    loss = xx.mean() + yy.mean() - 2 * xy.mean()
    return loss


# class for model training
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience):
        """
        Arg: patience (int): How long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_performance, model):

        if self.best_score is None:
            self.best_score = val_performance
            self.save_checkpoint(model)
        
        elif val_performance < (self.best_score)*1.0001 :
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_performance
            self.save_checkpoint(model)
            self.counter = 0
    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'checkpoint.pt')

class MINE(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        joint = torch.cat([x, y], dim=1)
        marg = torch.cat([x, y[torch.randperm(y.size(0))]], dim=1)

        t_joint = self.net(joint)
        t_marg = self.net(marg)

        mi_est = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marg)) + 1e-6)
        return -mi_est  # Negative since we minimize loss
            
def cross_covariance_loss(x, y):
    #x = F.softmax(x, dim=1)
    #y = F.softmax(y, dim=1)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    
    N = x.shape[0]  # Number of samples
    cov_xy = (x.T @ y) / N  # Compute covariance matrix
    
    return torch.sum(cov_xy ** 2)  # Minimize squared covariance terms

class UnifiedDataLoader:
    """Handles different data loading patterns"""
    
    @staticmethod
    def extract_batch_data(batch, training_mode):
        """Extract data from batch based on training mode"""
        if training_mode == "graph":
            return {
                'text': batch.input_ids,
                'categorical': batch.categoricals,
                'numerical': batch.numericals,
                'y': batch.y,
                'mask': batch.attention_masks,
                'edge_index': batch.edge_index,
                'edge_attr': batch.edge_attr,
                'train_mask': getattr(batch, 'train_mask', None),
                'val_mask': getattr(batch, 'val_mask', None),
                'test_mask': getattr(batch, 'test_mask', None)
            }
        elif training_mode == "contrastive":
            anchor_batch, anchor_label = batch[0]
            p_batch, _ = batch[1]
            n_batch, _ = batch[2]
            
            return {
                'anchor': {
                    'text': anchor_batch[0],
                    'categorical': anchor_batch[1],
                    'numerical': anchor_batch[2],
                    'mask': anchor_batch[3],
                    'y': anchor_label
                },
                'positive': {
                    'text': p_batch[0],
                    'categorical': p_batch[1],
                    'numerical': p_batch[2],
                    'mask': p_batch[3]
                },
                'negative': {
                    'text': n_batch[0],
                    'categorical': n_batch[1],
                    'numerical': n_batch[2],
                    'mask': n_batch[3]
                }
            }
        else:  # standard
            return {
                'text': batch[0],
                'categorical': batch[1],
                'numerical': batch[2],
                'y': batch[3],
                'mask': batch[4]
            }

class UnifiedLossCalculator:
    """Handles different loss calculation strategies"""
    
    def __init__(self, model_type, device):
        self.model_type = model_type
        self.device = device
        self.mine_network = None
        self.mine_optimizer = None
        self.triplet_loss = TripletLoss(margin=0.1)
        
        if "MINE" in model_type:
            self.mine_network = MINE(input_dim=256, hidden_dim=64).to(device)
            self.mine_optimizer = torch.optim.Adam(self.mine_network.parameters(), lr=1e-4)
    
    def calculate_loss(self, outputs, targets, criterion, additional_outputs=None):
        """Calculate loss based on model type"""
        
        if self.model_type == "LANISTR":
            return criterion(outputs, targets)
        
        elif any(loss_type in self.model_type for loss_type in ["MINE", "InfoNCE", "MMD"]):
            y_hat, bert_outputs, mlp_outputs = outputs
            comb_loss = criterion(y_hat, targets)
            
            if "MINE" in self.model_type:
                if self.mine_optimizer:
                    self.mine_optimizer.zero_grad()
                bert_outputs = bert_outputs.detach()
                additional_loss = self.mine_network(bert_outputs, mlp_outputs)
                if self.mine_optimizer:
                    self.mine_optimizer.step()
                    
            elif "InfoNCE" in self.model_type:
                additional_loss = info_nce_loss(bert_outputs, mlp_outputs, temperature=0.5)
                
            elif "MMD" in self.model_type:
                additional_loss = mmd_loss(bert_outputs, mlp_outputs, sigma=10.0)
            
            loss = comb_loss + additional_loss
            
        else:  # Standard models
            loss = criterion(outputs, targets)
        
        # Add regularization if enabled
        if L1:
            lambda_l1 = 1e-7 if "MINE" not in self.model_type else 1e-4
            l1_norm = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
            loss += lambda_l1 * l1_norm
            
        if L2:
            l2_lambda = 1e-4
            l2_reg = sum(param.pow(2).sum() for param in self.model.parameters())
            loss += l2_lambda * l2_reg
            
        return loss
    
    def calculate_contrastive_loss(self, anchor_outputs, positive_outputs, negative_outputs, targets, criterion):
        """Calculate contrastive loss for triplet training"""
        classification_loss = criterion(anchor_outputs, targets)
        triplet_loss_val = self.triplet_loss(anchor_outputs, positive_outputs, negative_outputs)
        return classification_loss + triplet_loss_val

class UnifiedModelForward:
    """Handles different model forward passes"""
    
    @staticmethod
    def forward_pass(model, model_type, data, training_mode="standard"):
        """Unified forward pass for different model types and training modes"""
        
        if training_mode == "graph":
            return model(
                data['text'], data['mask'], data['categorical'], 
                data['numerical'].float(), data['edge_index'], data['edge_attr']
            )
        elif training_mode == "contrastive":
            anchor_data = data['anchor']
            positive_data = data['positive']
            negative_data = data['negative']
            
            anchor_out = model(
                anchor_data['text'], anchor_data['mask'], 
                anchor_data['categorical'], anchor_data['numerical'].float()
            )
            positive_out = model(
                positive_data['text'], positive_data['mask'],
                positive_data['categorical'], positive_data['numerical'].float()
            )
            negative_out = model(
                negative_data['text'], negative_data['mask'],
                negative_data['categorical'], negative_data['numerical'].float()
            )
            
            return anchor_out, positive_out, negative_out
        else:  # standard
            return model(
                data['text'], data['mask'], data['categorical'], data['numerical']
            )

class UnifiedTrainer:
    """Unified training class that handles all training variants"""
    
    def __init__(self, model, model_type, dataset, device, training_mode="standard"):
        self.model = model
        self.model_type = model_type
        self.dataset = dataset
        self.device = device
        self.training_mode = training_mode
        self.loss_calculator = UnifiedLossCalculator(model_type, device)
        
    def train_epoch(self, loader, criterion, optimizer, seed):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        torch.manual_seed(seed)
        
        for batch in loader:
            optimizer.zero_grad()
            
            # Extract batch data
            data = UnifiedDataLoader.extract_batch_data(batch, self.training_mode)
            
            # Move data to device
            data = self._move_to_device(data)
            
            # Forward pass
            if self.training_mode == "contrastive":
                anchor_out, positive_out, negative_out = UnifiedModelForward.forward_pass(
                    self.model, self.model_type, data, self.training_mode
                )
                
                # Handle different output formats
                if isinstance(anchor_out, tuple):
                    anchor_out = anchor_out[0]
                    positive_out = positive_out[0]
                    negative_out = negative_out[0]
                
                loss = self.loss_calculator.calculate_contrastive_loss(
                    anchor_out, positive_out, negative_out, data['anchor']['y'], criterion
                )
                batch_size = data['anchor']['y'].shape[0]
                
            else:
                outputs = UnifiedModelForward.forward_pass(
                    self.model, self.model_type, data, self.training_mode
                )
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:  # Combined model with bert/mlp outputs
                        loss = self.loss_calculator.calculate_loss(outputs, data['y'], criterion)
                    else:  # Standard tuple output
                        loss = self.loss_calculator.calculate_loss(outputs[0], data['y'], criterion)
                else:
                    loss = self.loss_calculator.calculate_loss(outputs, data['y'], criterion)
                
                batch_size = data['y'].shape[0]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record loss
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        
        return total_loss / total_samples
    
    def evaluate(self, loader, criterion, seed):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        torch.manual_seed(seed)
        
        with torch.no_grad():
            for batch in loader:
                data = UnifiedDataLoader.extract_batch_data(batch, self.training_mode)
                data = self._move_to_device(data)
                
                outputs = UnifiedModelForward.forward_pass(
                    self.model, self.model_type, data, self.training_mode
                )
                
                if isinstance(outputs, tuple):
                    loss = criterion(outputs[0], data['y'])
                else:
                    loss = criterion(outputs, data['y'])
                
                batch_size = data['y'].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return total_loss / total_samples
    
    def compute_performance(self, loader, seed, data_split=""):
        """Compute performance metrics"""
        self.model.eval()
        preds_list = []
        labels_list = []
        
        torch.manual_seed(seed)
        
        with torch.no_grad():
            for batch in loader:
                data = UnifiedDataLoader.extract_batch_data(batch, self.training_mode)
                data = self._move_to_device(data)
                
                outputs = UnifiedModelForward.forward_pass(
                    self.model, self.model_type, data, self.training_mode
                )
                
                if self.training_mode == "graph":
                    pred = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    # Apply masks for graph data
                    if "train" in data_split:
                        pred = pred[data['train_mask']]
                        labels = data['y'][data['train_mask']]
                    elif "val" in data_split:
                        pred = pred[data['val_mask']]
                        labels = data['y'][data['val_mask']]
                    elif "test" in data_split:
                        pred = pred[data['test_mask']]
                        labels = data['y'][data['test_mask']]
                    else:
                        pred = pred
                        labels = data['y']
                        
                    p_hat = F.softmax(pred, dim=1)
                    
                else:
                    pred = outputs[0] if isinstance(outputs, tuple) else outputs
                    p_hat = F.softmax(pred, dim=1)
                    
                    labels = data['y']
                
                if p_hat.size(0) > 0:  # Check if there are samples
                    preds_list.append(p_hat)
                    labels_list.append(labels)
        
        if not preds_list:
            return {"accuracy": None, "micro_f1": None, "macro_f1": None, "auc": None}
        
        labels_list = torch.cat(labels_list)
        preds_list = torch.cat(preds_list)
        preds_classes = torch.argmax(preds_list, dim=1)
        
        # Calculate metrics
        accuracy = (preds_classes == labels_list).float().mean().item()
        
        # Move to CPU for sklearn
        preds_classes_np = preds_classes.cpu().numpy()
        labels_np = labels_list.cpu().numpy()
        preds_probs_np = preds_list.cpu().numpy()
        
        # Micro and Macro F1
        micro_f1 = f1_score(labels_np, preds_classes_np, average='micro')
        macro_f1 = f1_score(labels_np, preds_classes_np, average='macro')
        
        # AUC only for binary classification
        auc = None
        if preds_probs_np.shape[1] == 2:
            auc = roc_auc_score(labels_np, preds_probs_np[:, 1])
        
        return {
            "accuracy": accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "auc": auc,
        }
    
    def train(self, loader_train, loader_validation, n_epochs, criterion, optimizer, 
              factor=0.95, seed=42, verbose=True, triplet_loader=None):
        """Main training loop"""
        
        # Use triplet loader if provided and training mode is contrastive
        if triplet_loader is not None:
            self.training_mode = "contrastive"
            train_loader = triplet_loader
        else:
            train_loader = loader_train
        
        best_val_perf = 0
        scheduler = ExponentialLR(optimizer, gamma=factor)
        
        training_loss = []
        validation_loss = []
        training_acc = []
        validation_acc = []
        
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, criterion, optimizer, seed)
            
            # Validation
            val_loss = self.evaluate(loader_validation, criterion, seed)
            
            # Performance computation
            train_perf = self.compute_performance(loader_train, seed, f"{self.dataset}_train")["accuracy"]
            val_perf = self.compute_performance(loader_validation, seed, f"{self.dataset}_val")["accuracy"]
            
            # Scheduler step
            scheduler.step()
            
            # Record metrics
            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            training_acc.append(train_perf)
            validation_acc.append(val_perf)
            
            if verbose:
                end_time = time.time()
                print(f"---------training time (s): {round(end_time-start_time,0)} ---------")
                print(f"epoch: {epoch}, training loss: {round(train_loss,5)}")
                print(f"epoch: {epoch}, validation loss: {round(val_loss,5)}, validation performance: {round(val_perf,5)}")
            
            # Save best model
            if val_perf > best_val_perf * 1.001:
                torch.save(self.model, 'checkpoint.pt')
                best_val_perf = val_perf
            else:
                break
        
        # Load best model
        self.model = torch.load("checkpoint.pt", weights_only=False)
        
        # Save training history if needed
        self._save_training_history(training_loss, validation_loss, training_acc, validation_acc, seed)
        
        return self.model, epoch
    
    def _move_to_device(self, data):
        """Move data to device recursively"""
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data
    
    def _save_training_history(self, training_loss, validation_loss, training_acc, validation_acc, seed=42):
        """Save training history if enabled"""
        save_ = False  # Set to True if you want to save
        
        if save_:
            directory = f'Outputs_{VERSION}'
            os.makedirs(directory, exist_ok=True)
            
            files = {
                f"training_loss_{self.model_type}_{self.dataset}_{seed}.pkl": training_loss,
                f"validation_loss_{self.model_type}_{self.dataset}_{seed}.pkl": validation_loss,
                f"training_accuracy_{self.model_type}_{self.dataset}_{seed}.pkl": training_acc,
                f"validation_accuracy_{self.model_type}_{self.dataset}_{seed}.pkl": validation_acc,
            }
            
            for filename, data in files.items():
                filepath = os.path.join(directory, filename)
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)

# Convenience functions for backward compatibility
def unified_training(model, dataset, model_type, loader_train, n_epochs, loader_validation, 
                    criterion, optimizer, factor=0.95, seed=42, verbose=True, device="cuda",
                    training_mode="standard", triplet_loader=None):
    """
    Unified training function that replaces all the individual training functions
    
    Args:
        model: PyTorch model
        dataset: Dataset name
        model_type: Type of model (affects loss calculation)
        loader_train: Training data loader
        n_epochs: Number of epochs
        loader_validation: Validation data loader
        criterion: Loss criterion
        optimizer: Optimizer
        factor: Learning rate decay factor
        seed: Random seed
        verbose: Whether to print progress
        device: Device to use
        training_mode: "standard", "graph", or "contrastive"
        triplet_loader: Triplet loader for contrastive learning (optional)
    
    Returns:
        Trained model and final epoch number
    """
    trainer = UnifiedTrainer(model, model_type, dataset, device, training_mode)
    return trainer.train(loader_train, loader_validation, n_epochs, criterion, optimizer, 
                        factor, seed, verbose, triplet_loader)

def unified_evaluation(model_type, model, loader, criterion, seed, device, training_mode="standard"):
    """
    Unified evaluation function
    """
    trainer = UnifiedTrainer(model, model_type, "", device, training_mode)
    return trainer.evaluate(loader, criterion, seed)

def unified_performance(model, dataset, loader_target, model_type, seed, device, 
                       training_mode="standard", data_split=""):
    """
    Unified performance computation function
    """
    trainer = UnifiedTrainer(model, model_type, dataset, device, training_mode)
    return trainer.compute_performance(loader_target, seed, data_split)



# model optimization and selection
def hp_optimization_pretrained(model_type, criterion, seed, device, datasets_names, all_datasets, experiment_type="general"):
    """
    Hyperparameter optimization for pretrained models.
    
    Args:
        experiment_type: "general" for general hyperparameter tuning, "CA" for CrossAttention specific tuning
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # settings
    EPOCHS = 5
    
    # define model
    def define_model(trial, num_cat_var, num_numerical_var, cat_vocab_sizes, N_CLASSES):
        
        D_MODEL = 768
        N_LAYERS = 4

        if ["Concat4s", "sumW4s", "Concat2s", "sumW2s"].count(model_type) > 0:
            ATTENTION = 1
        else:
            ATTENTION = 0

        if experiment_type == "general":
            CA_DROPOUT = 0.5
            D_FF = 768
            DROPOUT = trial.suggest_float("DROPOUT", 0.1, 0.3, step = 0.1)
            D_FC = trial.suggest_int("D_FC", 64, 256, step = 64)
            N_HEADS = 8
        elif experiment_type == "CA":
            D_FF = trial.suggest_int("D_FF", 256, 768, step = 256)
            N_HEADS = trial.suggest_int("N_HEADS", 4, 8, step = 4)
            CA_DROPOUT = trial.suggest_categorical('CA_DROPOUT', [0.1, 0.5, 0.9])
            D_FC = 128
            DROPOUT = 0.1

        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='linear')
        if "Fourier" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='fourier')
        elif "PosEnVec" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='positional')
        elif "FourierVec" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='fourier_vec')
        elif "RBF" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='rbf')
        elif "RBFVec" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='rbf_vec')
        elif "Sigmoid" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='sigmoid')
        elif "Chebyshev" in model_type:    
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='chebyshev')
        
        if "CombinedModel" in model_type:
            MultiModelsObj = UniModels
        elif "CrossAttention" in model_type:
            MultiModelsObj = CrossAttention
        elif "CrossAttentionSkipNet" in model_type:
            MultiModelsObj = CrossAttentionSkipNet

        if model_type == "BertWithTabular":
            torch.manual_seed(seed)
            model = BertWithTabular(num_cat_var,
                                 num_numerical_var,
                                 N_HEADS,
                                 N_LAYERS,
                                 cat_vocab_sizes, 
                                 N_CLASSES,
                                 DROPOUT,
                                 D_MODEL,
                                 device)

        if model_type == "TabularForBert":
            torch.manual_seed(seed)
            model = TabularForBert(num_cat_var, 
                                    num_numerical_var, 
                                    N_HEADS, 
                                    N_LAYERS, 
                                    cat_vocab_sizes, 
                                    N_CLASSES, 
                                    DROPOUT, 
                                    D_MODEL, 
                                    device)
    
        if model_type == "OnlyTabular":
            torch.manual_seed(seed)
            model = OnlyTabular(num_cat_var, 
                                    num_numerical_var, 
                                    N_HEADS, 
                                    N_LAYERS, 
                                    cat_vocab_sizes, 
                                    N_CLASSES, 
                                    DROPOUT, 
                                    device)
            
        if model_type == "OnlyText":
            torch.manual_seed(seed)
            model = OnlyText(num_cat_var, 
                            num_numerical_var, 
                            N_HEADS, 
                            N_LAYERS, 
                            cat_vocab_sizes, 
                            N_CLASSES, 
                            DROPOUT, 
                            D_MODEL, 
                            device)
                            
        if model_type == "CombinedModelGAT":
            torch.manual_seed(seed)
            model = CombinedModelGAT(num_cat_var, 
                            num_numerical_var, 
                            N_HEADS, 
                            N_LAYERS, 
                            cat_vocab_sizes, 
                            N_CLASSES, 
                            DROPOUT, 
                            D_MODEL, 
                            device)
            
        if model_type[-5:] == "SumW2":
            torch.manual_seed(seed)
            model = CombinedModelSumW2(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
            
        if model_type[-7:] == "Concat2":
            torch.manual_seed(seed)
            model = CombinedModelConcat2(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
                                        
        if "SumW4" in model_type:
            torch.manual_seed(seed)
            model = CombinedModelSumW4(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
        
        if "Concat4" in model_type:
            torch.manual_seed(seed)
            model = CombinedModelConcat4(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
                         
        return model
        
    def objective(trial):
    
      dataset_names = datasets_names  # List of dataset names or identifiers
      all_accuracies = []
      
      if experiment_type == "general":
          REGULARIZATION = trial.suggest_categorical("REGULARIZATION", [0, 1, 2])
          LR = trial.suggest_categorical('LR', [1e-5, 5e-5, 1e-4, 5e-4])
      elif experiment_type == "CA":
          REGULARIZATION = 0
          LR = 5e-4
  
      for dataset_name in dataset_names:
          # Load dataset-specific settings
          categorical_var = all_datasets[dataset_name]["categorical_var"]
          numerical_var = all_datasets[dataset_name]["numerical_var"]
          cat_vocab_sizes = all_datasets[dataset_name]["cat_vocab_sizes"]
          N_CLASSES = all_datasets[dataset_name]["N_CLASSES"]
          WEIGHT_DECAY = all_datasets[dataset_name]["WEIGHT_DECAY"]
          CRITERION = all_datasets[dataset_name]["CRITERION"]
  
          # Redefine model for this dataset
          model = define_model(trial, numerical_var, categorical_var, cat_vocab_sizes, N_CLASSES).to(device)
          optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
          BATCH_SIZE = 32
          dataset_train = all_datasets[dataset_name]["train"]
          dataset_val = all_datasets[dataset_name]["val"]
          loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
          loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
  
          # Adjust number of training samples
          N_TRAIN_EXAMPLES = min(len(dataset_train), int(len(dataset_train)*0.6 if len(dataset_train) > 20000 else len(dataset_train)))
          N_TRAIN_EXAMPLES = (N_TRAIN_EXAMPLES // BATCH_SIZE) * BATCH_SIZE
  
          for epoch in range(EPOCHS):
              model.train()
              for batch_idx, batch in enumerate(loader_train):
                  if batch_idx * BATCH_SIZE >= N_TRAIN_EXAMPLES:
                      break
  
                  text, cat, num, y, mask = [x.to(device) for x in batch]
                  optimizer.zero_grad()
  
                  if model_type not in ["???"]:
                      y_hat = model(text, mask, cat, num)[0]
                      loss = CRITERION(y_hat, y)
                  else:
                      y_hat, bert_out, mlp_out = model(text, mask, cat, num)
                      loss = (CRITERION(y_hat, y) + 0.1 * cross_covariance_loss(mlp_out, bert_out)) / 2
  
                  # Optional: regularization
                  if REGULARIZATION == 1:
                      loss += 1e-4 * sum(torch.sum(torch.abs(p)) for p in model.parameters())
                  elif REGULARIZATION == 2:
                      loss += 1e-4 * sum(p.pow(2).sum() for p in model.parameters())
  
                  loss.backward()
                  optimizer.step()
  
          # Evaluate accuracy for this dataset
          acc_train = unified_performance(model, dataset_name, loader_train, model_type, seed, device, training_mode="standard", data_split="_train")["accuracy"]
          acc_val = unified_performance(model, dataset_name, loader_val, model_type, seed, device, training_mode="standard", data_split="_validation")["accuracy"]
          accuracy = min(acc_train, acc_val)
          all_accuracies.append(accuracy)
  
      return sum(all_accuracies) / len(all_accuracies)

    search_space_ca = {
                     'CA_DROPOUT': [0.1, 0.5, 0.9],
                     "N_HEADS": [4, 8],
                     "D_FF": [256, 512, 768],
                 }
    search_space = {
                      'DROPOUT': [0.1, 0.2, 0.3],
                      'LR': [1e-5, 5e-5, 1e-4, 5e-4],
                      "D_FC": [64, 128, 256],
                      "REGULARIZATION": [0,1,2]
                      
                  }
    if experiment_type == "general":
        sampler = optuna.samplers.GridSampler(search_space)
        n_trials = 108
    elif experiment_type == "CA":
        sampler = optuna.samplers.GridSampler(search_space_ca)
        n_trials = 18
        
    study = optuna.create_study(direction="maximize", sampler = sampler)
    study.optimize(objective, n_trials=n_trials) 
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    # select best trial
    trials = study.trials
    return trials
    

def hp_optimization_losses(model_type, criterion, seed, device, datasets_names, all_datasets):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # settings
    EPOCHS = 5

    #FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT= load_settings(dataset)
    
    # define model
    def define_model(trial, num_cat_var, num_numerical_var, cat_vocab_sizes, N_CLASSES):

        D_MODEL = 768
        N_LAYERS = 4
        N_HEADS = 8 #optuna CA best
        DROPOUT = 0.2
        CA_DROPOUT = 0.1 #CA optuna best
        ATTENTION = 0 
        D_FC = 256
        D_FF = 256 #best of optuna CA

        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='linear')
        if "Fourier" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='fourier')#d_ff
        elif "PosEnVec" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='positional')
        elif "FourierVec" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='fourier_vec')
        elif "RBF" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='rbf')
        elif "RBFVec" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='rbf_vec')
        elif "Sigmoid" in model_type:
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='sigmoid')
        elif "Chebyshev" in model_type:    
            TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, D_MODEL, numerical_transform='chebyshev')
        
        if "CombinedModel" in model_type:
            MultiModelsObj = UniModels
        elif "CrossAttention" in model_type:
            MultiModelsObj = CrossAttention
        elif "CrossAttentionSkipNet" in model_type:
            MultiModelsObj = CrossAttentionSkipNet
        

        if model_type == "BertWithTabular":
            torch.manual_seed(seed)
            model = BertWithTabular(num_cat_var,
                                 num_numerical_var,
                                 N_HEADS,
                                 N_LAYERS,
                                 cat_vocab_sizes, 
                                 N_CLASSES,
                                 DROPOUT,
                                 D_MODEL,
                                 device)

        if model_type == "TabularForBert":
            torch.manual_seed(seed)
            model = TabularForBert(num_cat_var, 
                                    num_numerical_var, 
                                    N_HEADS, 
                                    N_LAYERS, 
                                    cat_vocab_sizes, 
                                    N_CLASSES, 
                                    DROPOUT, 
                                    D_MODEL, 
                                    device)
    
        if model_type == "OnlyTabular":
            torch.manual_seed(seed)
            model = OnlyTabular(num_cat_var, 
                                    num_numerical_var, 
                                    N_HEADS, 
                                    N_LAYERS, 
                                    cat_vocab_sizes, 
                                    N_CLASSES, 
                                    DROPOUT, 
                                    device)
            
        if model_type == "OnlyText":
            torch.manual_seed(seed)
            model = OnlyText(num_cat_var, 
                            num_numerical_var, 
                            N_HEADS, 
                            N_LAYERS, 
                            cat_vocab_sizes, 
                            N_CLASSES, 
                            DROPOUT, 
                            D_MODEL, 
                            device)
                            
        if model_type == "CombinedModelGAT":
            torch.manual_seed(seed)
            model = CombinedModelGAT(num_cat_var, 
                            num_numerical_var, 
                            N_HEADS, 
                            N_LAYERS, 
                            cat_vocab_sizes, 
                            N_CLASSES, 
                            DROPOUT, 
                            D_MODEL, 
                            device)
            
        if model_type[-5:] == "SumW2":
            torch.manual_seed(seed)
            model = CombinedModelSumW2(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
            
        if model_type[-7:] == "Concat2":
            torch.manual_seed(seed)
            model = CombinedModelConcat2(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
                                        
        if "SumW4" in model_type:
            torch.manual_seed(seed)
            model = CombinedModelSumW4(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
        
        if "Concat4" in model_type:
            torch.manual_seed(seed)
            model = CombinedModelConcat4(MultiModelsObj,
                                        TabEmbed,
                                        num_cat_var, 
                                        num_numerical_var, 
                                        N_HEADS, 
                                        N_LAYERS, 
                                        cat_vocab_sizes, 
                                        N_CLASSES, 
                                        DROPOUT, 
                                        D_MODEL, 
                                        device,
                                        ATTENTION,
                                        D_FF,
                                        D_FC,
                                        CA_DROPOUT)
                                 
        return model
        
    def objective(trial):
    
      dataset_names = datasets_names  # List of dataset names or identifiers
      all_accuracies = []
  
      for dataset_name in dataset_names:
          # Load dataset-specific settings
          categorical_var = all_datasets[dataset_name]["categorical_var"]
          numerical_var = all_datasets[dataset_name]["numerical_var"]
          cat_vocab_sizes = all_datasets[dataset_name]["cat_vocab_sizes"]
          N_CLASSES = all_datasets[dataset_name]["N_CLASSES"]
          WEIGHT_DECAY = all_datasets[dataset_name]["WEIGHT_DECAY"]
          CRITERION = all_datasets[dataset_name]["CRITERION"]
  
          # Redefine model for this dataset
          model = define_model(trial, numerical_var, categorical_var, cat_vocab_sizes, N_CLASSES).to(device)
          LR = 1e-4
          if "InfoNCE" in model_type:
              tau = trial.suggest_categorical('tau', [0.05, 0.1, 0.2, 0.5])# InfoNCE temperature τ
          if "Contrastive" in model_type:
              alpha = trial.suggest_categorical('alpha', [0.1, 0.5, 1.0])# Triplet margin α 
          if "MMD" in model_type:
              sigma = trial.suggest_categorical('sigma', [1.0, 5.0, 10.0])# MMD kernel bandwidth σ
          if "MINE" in model_type:
              hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])# MINE hidden size
          optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
          BATCH_SIZE = 32
          dataset_train = all_datasets[dataset_name]["train"]
          dataset_val = all_datasets[dataset_name]["val"]
          loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
          loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
          if "Contrastive" in model_type:
              triplet_dataset = TripletDataset(loader_train)
              triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)
              triplet_loss = TripletLoss(margin=alpha)
          if "MINE" in model_type:
              mine_loss = MINE(input_dim=256, hidden_dim=hidden_size).to(device)
              mine_optimizer = torch.optim.Adam(mine_loss.parameters(), lr=1e-4)
  
          # Adjust number of training samples
          N_TRAIN_EXAMPLES = min(len(dataset_train), int(len(dataset_train)*0.6 if len(dataset_train) > 20000 else len(dataset_train)))
          N_TRAIN_EXAMPLES = (N_TRAIN_EXAMPLES // BATCH_SIZE) * BATCH_SIZE
          if "Contrastive" not in model_type:
              for epoch in range(EPOCHS):
                  model.train()
                  for batch_idx, batch in enumerate(loader_train):
                      if batch_idx * BATCH_SIZE >= N_TRAIN_EXAMPLES:
                          break
      
                      text, cat, num, y, mask = [x.to(device) for x in batch]
                      optimizer.zero_grad()
      
                      if model_type in ["???"]:
                          y_hat = model(text, mask, cat, num)[0]
                          loss = CRITERION(y_hat, y)
                      else:
                          y_hat, bert_outputs, mlp_outputs = model(text, mask, cat, num)
                          if "MINE" in model_type:
                              mine_optimizer.zero_grad()
                              bert_outputs = bert_outputs.detach()  # stop gradients to main encoder
                              mlp_outputs = mlp_outputs.detach()
                              
                              bert_outputs = bert_outputs.to(device)
                              mlp_outputs = mlp_outputs.to(device)
                              l = mine_loss(bert_outputs, mlp_outputs)
                              # Optimize MINE network
                              
                          elif "InfoNCE" in model_type:
                              l = info_nce_loss(bert_outputs, mlp_outputs, temperature=tau)
                          elif "MMD" in model_type: 
                              l = mmd_loss(bert_outputs, mlp_outputs, sigma=sigma)
                          loss = CRITERION(y_hat, y) + l
      
      
                      loss.backward()
                      optimizer.step()
                      if "MINE" in model_type:
                          #mine_loss.backward()
                          mine_optimizer.step()
          else:
              for epoch in range(EPOCHS):
                  start=time.time()
                  train_loss = 0 # training loss by sample
                  total = 0 # number of samples
                  torch.manual_seed(seed)
                  for anchor, positive, negative in triplet_loader:
                      model.train()
                      anchor_batch, anchor_label = anchor
                      p_batch, _ = positive
                      n_batch, _ = negative
                      
                      text = anchor_batch[0]
                      categorical = anchor_batch[1]
                      numerical = anchor_batch[2]
                      y = anchor_label
                      mask = anchor_batch[3]
                      
                      p_text = p_batch[0]
                      p_categorical = p_batch[1]
                      p_numerical = p_batch[2]
                      p_mask = p_batch[3]
                      
                      n_text = n_batch[0]
                      n_categorical = n_batch[1]
                      n_numerical = n_batch[2]
                      n_mask = n_batch[3]
                      
                      # 1. clear gradients
                      optimizer.zero_grad()
                      
                      # 2. to device
                      text = text.to(device)
                      mask = mask.to(device)
                      categorical = categorical.to(device)
                      numerical = numerical.to(device)
                      y = y.to(device)
                      
                      n_text = n_text.to(device)
                      n_mask = n_mask.to(device)
                      n_categorical = n_categorical.to(device)
                      n_numerical = n_numerical.to(device)
                      
                      p_text = p_text.to(device)
                      p_mask = p_mask.to(device)
                      p_categorical = p_categorical.to(device)
                      p_numerical = p_numerical.to(device)
                      
                      # 3. forward pass and compute loss
                      
                      y_hat = model(text, mask, categorical, numerical.float())[0]
                      p_y_hat = model(p_text, p_mask, p_categorical, p_numerical.float())[0]
                      n_y_hat = model(n_text, n_mask, n_categorical, n_numerical.float())[0]
                      loss = criterion(y_hat,y) +  triplet_loss(y_hat, p_y_hat, n_y_hat)
                      loss.backward()
                      optimizer.step()
  
          # Evaluate accuracy for this dataset
        #   acc_train = performance_pretrained(model, "_train", loader_train, model_type, seed, device)["accuracy"]
        #   acc_val = performance_pretrained(model, "_validation", loader_val, model_type, seed, device)["accuracy"]
          acc_train = unified_performance(model, all_datasets[dataset_name], loader_train, model_type, seed, device, training_mode="standard")["accuracy"]
          acc_val = unified_performance(model, all_datasets[dataset_name], loader_val, model_type, seed, device, training_mode="standard")["accuracy"]
          accuracy = min(acc_train, acc_val)
          all_accuracies.append(accuracy)
  
      return sum(all_accuracies) / len(all_accuracies)

    if "InfoNCE" in model_type:
        search_space = {'tau': [0.05, 0.1, 0.2, 0.5]}# InfoNCE temperature τ
    if "Contrastive" in model_type:
        search_space = {'alpha': [0.1, 0.5, 1.0]}# Triplet margin α 
    if "MMD" in model_type:
        search_space = {"sigma": [1.0, 5.0, 10.0]} # MMD kernel bandwidth σ
    if "MINE" in model_type:
        search_space = {"hidden_size": 	[64, 128, 256]} # MINE hidden size
    
    sampler = optuna.samplers.GridSampler(search_space)#search_space_ca
    study = optuna.create_study(direction="maximize", sampler = sampler)

    study.optimize(objective, n_trials=3) # 10 or 15 minutes, CA: n_trials=18
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    # select best trial
    trials = study.trials

    return trials