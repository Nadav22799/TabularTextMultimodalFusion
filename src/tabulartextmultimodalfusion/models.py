import torch
import torch.nn as nn
import numpy as np
import math
from transformers import BertModel, DistilBertModel, TrainingArguments
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv

# Positional encoding (as per "Attention is all you need")
def positional_embeddings(seq_len, d):
    result = torch.ones(seq_len, d)
    for i in range(seq_len):
        for j in range(d):
            result[i][j] = np.sin(i/(10000**(j/d))) if j%2 == 0 else np.cos(i/(10000**((j-1)/d))) 
    return result

#######################################################################################################################
# 5. Tabular-Text Transformer or TTT
#######################################################################################################################

## MM encoder block of the MMTransformer
class MMEncoderLayer2(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # hidden dimension in feedforward network
        self.dropout = dropout
        self.attention_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)
        
        # layer normalization
        self.lnorm1 = nn.LayerNorm(self.d_model)
        self.lnorm2 = nn.LayerNorm(self.d_model)
        
        # multi-head attention:
        self.ma = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads , dropout=self.dropout, batch_first=True)
        
        # FF layer
        self.ff1 = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_ff, self.d_model))
        
    def forward(self, source, target):
        
        # computing one-versus-all attention
        if type(source)!=list:
            target_txt = target[0]
            padding_mask = target[1]
            source = torch.cat([source, target_txt], dim = 1)
            target_cma = self.ma(query = self.lnorm1(target_txt), key = self.lnorm1(source), value = self.lnorm1(source), key_padding_mask = padding_mask)[0]
            target_out = target_txt + self.attention_dropout(target_cma)
        else:
            source_txt = source[0]
            padding_mask = source[1]
            source_txt = torch.cat([target, source_txt], dim = 1)
            target_cma = self.ma(query = self.lnorm1(target), key = self.lnorm1(source_txt), value = self.lnorm1(source_txt), key_padding_mask = padding_mask)[0]
            target_out = target + self.attention_dropout(target_cma)
            
        # layer norm, feed forward network, and residual connection
        target_out = target_out + self.output_dropout(self.ff1(self.lnorm2(target_out)))
        
        return target_out

    

class TTT(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 cat_vocab_sizes, num_cat_var,
                 num_numerical_var, quantiles,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes, device):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.quantiles = quantiles # quantiles for each numerical variable
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        self.device = device
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
           
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, 
                                            padding_idx = 0) for i in range(self.num_cat_var)])
    
        # embeddings for numericals
        self.num_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=len(self.quantiles[i]), embedding_dim=self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.text_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.text_pos_embed = nn.Parameter(data = positional_embeddings(self.max_len + 1, self.d_model), requires_grad = False)
         
        # MM encoder block
        self.MMEncoder_tab_txt = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        self.MMEncoder_txt_tab = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        
        # text fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # tabular fully connected network
        self.fc2 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        for i in range(self.num_numerical_var):
            nn.init.kaiming_uniform_(self.num_embeddings[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
        nn.init.zeros_(self.fc2[0].bias)
        nn.init.kaiming_uniform_(self.fc2[0].weight)
        nn.init.zeros_(self.fc2[3].bias)
        nn.init.kaiming_uniform_(self.fc2[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):

        # 1. reshape categoricals and numericals for embeddings
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        C = torch.zeros((len(numericals), len(self.quantiles[0])), dtype = int).to(self.device)
        for i in range(len(self.quantiles[0])):
            C[:,i] = i
        numerical_token_list = [C for i in range(self.num_numerical_var)]
        weights = [np.abs(np.array([numericals[:,i].cpu().numpy() for j in range(len(self.quantiles[i]))]).T - self.quantiles[i]) for i in range(self.num_numerical_var)] # distance to quantile
        numericals = numericals.to(self.device) 
        weights = [torch.tensor(1/w).to(self.device)  for w in weights] # similarity
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [torch.nan_to_num(weights[i], 1.) for i in range(self.num_numerical_var)] # replace nan by 1
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [weights[i].unsqueeze(1) for i in range(self.num_numerical_var)]
        weights = [weights[i].float() for i in range(self.num_numerical_var)]
       
        # 2. embedding layers
        # text embedding
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        # categorical embedding
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        # numerical embedding
        num_embedding_list = [self.num_embeddings[i](numerical_token_list[i]) for i in range(self.num_numerical_var)]
        # numericals: Weights x Quantile Embeddings
        numericals = [torch.bmm(weights[i], num_embedding_list[i]) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numericals[i] for i in range(self.num_numerical_var)], dim=1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        texts = torch.stack([torch.vstack((self.text_cls, texts[i])) for i in range(len(texts))]) * math.sqrt(self.d_model)
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. positional embeddings
        text_pos_embeds = self.text_pos_embed.repeat(len(texts),1,1)
        texts = texts + text_pos_embeds

        ## 5. MM encoder
        texts_dict = {}
        texts_dict[0] = texts
        tabulars_dict = {}
        tabulars_dict[0] = tabulars
        for i, layer in enumerate(self.MMEncoder_tab_txt):
            texts_dict[i+1] = layer(source = tabulars_dict[0], target = [texts_dict[i], padding_mask])
        for i, layer in enumerate(self.MMEncoder_txt_tab):
            tabulars_dict[i+1] = layer(source = [texts_dict[0], padding_mask], target = tabulars_dict[i])
        texts = texts_dict[i+1]
        tabulars = tabulars_dict[i+1]

        # 6. Extract CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        
        #7. Late fusion (average of logits)
        text_pred = self.fc1(text_cls)
        tabular_pred = self.fc2(tabular_cls)
        pred = (text_pred + tabular_pred)/2 # average
        
        return pred, text_pred, tabular_pred
            
#######################################################################################################################
# 9. LateFuseBERT
#######################################################################################################################

class LateFuseBERT(nn.Module):
    def __init__(self,
                 text_model,
                 cat_vocab_sizes,
                 num_cat_var,
                 num_numerical_var,
                 d_model,
                 n_heads,
                 n_layers, 
                 dropout,
                 d_fc,
                 n_classes):
        # super constructor
        super().__init__()

        # attributes
        self.text_model = text_model
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.n_heads = n_heads # number of attention heads
        self.d_model = d_model # embedding dimension
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, padding_idx = 0) for i in range(self.num_cat_var)])
    
        # linear mapper for numerical data
        self.num_linears = nn.ModuleList([nn.Linear(1, self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
          
        # Self Attention Transformer encoder
        self.tab_encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_model, dropout=self.dropout, batch_first=True)
        self.tab_transformer_encoder = nn.TransformerEncoder(self.tab_encoder_layers, self.n_layers)
        
        # last fully connected network
        self.fc1 = nn.Sequential(nn.Linear(2*self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()

    def init_weights(self):
        # embeddings
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        # numerical linear
        for i in range(self.num_numerical_var):
            nn.init.zeros_(self.num_linears[i].bias)
            nn.init.kaiming_uniform_(self.num_linears[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)


    def forward(self, texts, attention_mask, categoricals, numericals):

        # 1. reshape categoricals for embeddings and numericals before linear transformation 
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        numerical_list = [numericals[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(self.num_numerical_var)]
        
        # 2. embedding layers
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        numerical_embedding_list = [self.num_linears[i](numerical_list[i].float()) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numerical_embedding_list[i] for i in range(self.num_numerical_var)], dim = 1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. add classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. Self attention Transformer encoder (tabular stream)
        tabulars = self.tab_transformer_encoder(tabulars)
   
        # 5. text model prediction
        texts = self.text_model(texts, attention_mask = attention_mask).last_hidden_state

        # 6. Concatenate CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        mm_cls = torch.cat([text_cls, tabular_cls], dim = 1)

        # 7. Fully connected network for classification purpose
        pred = self.fc1(mm_cls)
        
        return pred, text_cls, tabular_cls

#######################################################################################################################
# 10. AllTextBERT
#######################################################################################################################

class AllTextBERT(nn.Module):
    def __init__(self,
                 text_model,
                 d_model,
                 dropout,
                 d_fc,
                 n_classes):
        # super constructor
        super().__init__()

        # attributes
        self.text_model = text_model
        self.d_model = d_model
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.dropout = dropout # dropout rate
        self.n_classes = n_classes # number of classes
        
        # last fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))

        
        # weight initialization
        self.init_weights()

    def init_weights(self):
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)

    def forward(self, texts, attention_mask, categoricals, numericals):

        # 1. text model prediction
        texts = self.text_model(texts, attention_mask = attention_mask).last_hidden_state

        # 2. Extract CLS tokens
        text_cls = texts[:,0,:]
        
        # 3. Logits
        text_pred = self.fc1(text_cls)

        return text_pred, text_cls
        
#########################################################################################################
#########################################################################################################
# MY MODELS
#########################################################################################################
#########################################################################################################

#######################################################################################################################
# TabularForBert
#######################################################################################################################  
class TabularForBert(nn.Module):
    def __init__(self, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device):
        super(TabularForBert, self).__init__()
        
        self.dropout = dropout # dropout rate
        self.num_cat_var = num_cat_var
        
        self.cat_vocab_sizes = cat_embed_dims
        self.d_model = d_model
        self.num_numerical_var = num_numerical_var
        
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, padding_idx = 0) for i in range(self.num_cat_var)])
    
        # linear mapper for numerical data
        self.num_linears = nn.ModuleList([nn.Linear(1, self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
          
        # Self Attention Transformer encoder
        self.tab_encoder_layers = nn.TransformerEncoderLayer(self.d_model, nhead, 
                                                              self.d_model, dropout=self.dropout, batch_first=True)
        self.tab_transformer_encoder = nn.TransformerEncoder(self.tab_encoder_layers, n_layers)
        
        self.fc = nn.Linear(self.d_model, num_classes)

    def forward(self, input_ids, attention_mask, categoricals, numericals):
        
        # 1. reshape categoricals for embeddings and numericals before linear transformation 
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        numerical_list = [numericals[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(self.num_numerical_var)]
        
        # 2. embedding layers
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        #categoricals = self.cat_dropout(categoricals)
        numerical_embedding_list = [self.num_linears[i](numerical_list[i].float()) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numerical_embedding_list[i] for i in range(self.num_numerical_var)], dim = 1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. add classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. Self attention Transformer encoder (tabular stream)
        tabulars = self.tab_transformer_encoder(tabulars)

        tabular_output = tabulars[:,0,:]

        logits = self.fc(tabular_output)

        return logits, None, tabular_output

#######################################################################################################################
# OnlyTabular
#######################################################################################################################
        
class OnlyTabular(nn.Module):
    def __init__(self, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, device):
        super(OnlyTabular, self).__init__()
        """
        OnlyTabular processes categorical and numerical features using an MLP model for classification.
        
        Args:
            num_cat_var (int): Number of categorical variables.
            num_numerical_var (int): Number of numerical variables.
            nhead (int): Not used in this model (reserved for future extensions).
            n_layers (int): Not used in this model (reserved for future extensions).
            cat_embed_dims (list): List containing embedding dimensions for each categorical variable.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
            device (str or torch.device): Device to run the model on.
        """
        
        self.dropout = dropout # dropout rate
        self.cat_vocab_sizes = cat_embed_dims
        self.d_model = sum(cat_embed_dims) + num_numerical_var# Model dimension for scaling #v1= sum(cat_embed_dims), v2=num_cat_var
        self.num_numerical_var = num_numerical_var
        
        # MLP model for tabular classification
        self.mlp = MLPModel(input_dim=self.d_model, num_classes=num_classes, dropout=self.dropout).to(device)#v1= MLPModel4, v2 = MLPModel5
        
    def forward(self, input_ids, attention_mask, categoricals, numericals):
        """
        Forward pass of the OnlyTabular model.
        
        Args:
            input_ids (torch.Tensor): Not used in this model (included for compatibility with multimodal settings).
            attention_mask (torch.Tensor): Not used in this model (included for compatibility with multimodal settings).
            categoricals (torch.Tensor): Categorical input features.
            numericals (torch.Tensor): Numerical input features.
        
        Returns:
            tuple: Contains logits and tabular output features.
        """
        
        # Process numerical features
        numerical_list = [numericals[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(self.num_numerical_var)]
        numerical_embedding_list = [numerical_list[i].float() for i in range(self.num_numerical_var)]
        numericals = torch.cat([numerical_embedding_list[i] for i in range(self.num_numerical_var)], dim = 1)
        numericals = numericals.squeeze(dim=2)
        
        # One-hot encoding for categorical variables
        one_hot_encoded = [F.one_hot(categoricals[:, i], num_classes=self.cat_vocab_sizes[i]) for i in range(len(self.cat_vocab_sizes))]
        one_hot_encoded = torch.cat(one_hot_encoded, dim=-1)
        categoricals = one_hot_encoded
        
        # Concatenate categorical and numerical features
        tabulars = torch.cat([categoricals, numericals], dim=1)     

        # 4. Extract [CLS] token from tabular transformer output
        tabular_output = self.mlp(tabulars)

        # Pass through MLP model
        logits = tabular_output

        return logits, None, tabular_output

#######################################################################################################################
# OnlyText
#######################################################################################################################

class OnlyText(nn.Module):
    def __init__(self, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device):
        super(OnlyText, self).__init__()
        
        # Bert Model
        self.bert = Bert().to(device)
        self.fc = nn.Linear(d_model, num_classes)
       
    def forward(self, input_ids, attention_mask, categoricals, numericals):

        # Process text inputs using BERT
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]# Pooled output
        logits = self.fc(bert_outputs)

        return logits, bert_outputs, None

#######################################################################################################################
# CombinedModelDgate1
####################################################################################################################### 
class CombinedModelDgate1(nn.Module):
    def __init__(self, MultiModelsObj, TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(CombinedModelDgate1, self).__init__()

        self.base_model = MultiModelsObj(TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout)
        self.fc = nn.Sequential(nn.Linear(d_model, d_fc),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_fc, num_classes))
            
        self.device = device

    def forward(self, input_ids, attention_mask, categoricals, numericals):

        pooled_output1, pooled_output2, input_embeddings1, input_embeddings2 = self.base_model(input_ids, attention_mask, categoricals, numericals)

        gating_output1 = torch.sigmoid(pooled_output2) 
        combined_output = gating_output1 * pooled_output1
        
        logits = self.fc(combined_output)

        return logits, None, None
    
#######################################################################################################################
# CombinedModelSumW2
####################################################################################################################### 
class CombinedModelSumW2(nn.Module):
    def __init__(self, MultiModelsObj, TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(CombinedModelSumW2, self).__init__()

        self.base_model = MultiModelsObj(TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout)
        self.weights = nn.Parameter(torch.randn(2, d_model))
        #self.fc = nn.Linear(d_model, num_classes)
        self.fc = nn.Sequential(nn.Linear(d_model, d_fc),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_fc, num_classes))
            
        self.device = device

    def forward(self, input_ids, attention_mask, categoricals, numericals):

        pooled_output1, pooled_output2, _, _ = self.base_model(input_ids, attention_mask, categoricals, numericals)

        combined_output = self.weights[0] * pooled_output1 + self.weights[1] * pooled_output2
        
        logits = self.fc(combined_output)

        return logits, None, None
        
#######################################################################################################################
# CombinedModelSumW4
####################################################################################################################### 
class CombinedModelSumW4(nn.Module):
    def __init__(self, MultiModelsObj, TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(CombinedModelSumW4, self).__init__()

        self.base_model = MultiModelsObj(TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout)
        self.weights = nn.Parameter(torch.randn(4, d_model))
        #self.weights = nn.Parameter(torch.ones(2, 1))
        #self.fc = nn.Linear(d_model, num_classes)
        self.fc1 = nn.Sequential(nn.Linear(d_model, d_fc),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_fc, num_classes))
        
        self.fc2 = nn.Sequential(nn.Linear(2 * d_model, d_fc),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_fc, num_classes))
                                
        
        self.device = device

    def forward(self, input_ids, attention_mask, categoricals, numericals):

        pooled_output1, pooled_output2, input_embeddings1, input_embeddings2 = self.base_model(input_ids, attention_mask, categoricals, numericals)

        combined_output = self.weights[0] * pooled_output1 + self.weights[1] * pooled_output2 + self.weights[2] * input_embeddings1 + self.weights[3] * input_embeddings2
        #weights = F.softmax(self.weights, dim=1)  # Normalize scores into probabilities
        
        #combined_output = weights[0] * self.fc1(torch.cat([pooled_output1, pooled_output2], dim=1)) + weights[1] * self.fc1(torch.cat([input_embeddings1, input_embeddings2], dim=1))
        #combined_output = weights[0] * torch.cat([pooled_output1, pooled_output2], dim=1) + weights[1] * torch.cat([input_embeddings1, input_embeddings2], dim=1)       
        
        logits = self.fc1(combined_output)

        return logits, None, None

#######################################################################################################################
# CombinedModelConcat2
####################################################################################################################### 
class CombinedModelConcat2(nn.Module):
    def __init__(self, MultiModelsObj, TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(CombinedModelConcat2, self).__init__()

        self.base_model = MultiModelsObj(TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout)
        #self.weights = nn.Parameter(torch.randn(2, d_model))
        #self.fc = nn.Linear(2 * d_model, num_classes)
        self.fc = nn.Sequential(nn.Linear(2 * d_model, d_fc), #best d_fc before hf is 128
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_fc, num_classes))
            
        self.device = device

    def forward(self, input_ids, attention_mask, categoricals, numericals):

        pooled_output1, pooled_output2, _, _ = self.base_model(input_ids, attention_mask, categoricals, numericals)

        #combined_output = self.weights[0] * pooled_output1 + self.weights[1] * pooled_output2
        
        combined_output = torch.cat([pooled_output1, pooled_output2], dim=1)
        
        logits = self.fc(combined_output)

        return logits, None, None

#######################################################################################################################
# CombinedModelConcat4
####################################################################################################################### 
class CombinedModelConcat4(nn.Module):
    def __init__(self, MultiModelsObj, TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(CombinedModelConcat4, self).__init__()

        self.base_model = MultiModelsObj(TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout)
        #self.weights = nn.Parameter(torch.randn(2, 1))
        #self.fc = nn.Linear(4 * d_model, num_classes)
        self.fc_ta = nn.Linear(d_model, num_classes)
        self.fc_te = nn.Linear(d_model, num_classes)
        self.fc = nn.Sequential(nn.Linear(4 * d_model, d_fc), #best d_fc before hf is 128
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_fc, num_classes))
        #self.dropout = nn.Dropout(dropout)# before hf 0.1 best in CA without self
        #self.num_numerical_var = num_numerical_var
            
        self.device = device

    def forward(self, input_ids, attention_mask, categoricals, numericals):

        pooled_output1, pooled_output2, input_embeddings1, input_embeddings2 = self.base_model(input_ids, attention_mask, categoricals, numericals)
        #input_embeddings1 = self.dropout(input_embeddings1)
        #input_embeddings2 = self.dropout(input_embeddings2)
        
        #weights = F.softmax(self.weights, dim=1)  # Normalize scores into probabilities

        #combined_output = self.weights[0] * pooled_output1 + self.weights[1] * pooled_output2
        combined_output = torch.cat([pooled_output1, pooled_output2, input_embeddings1, input_embeddings2], dim=1)
        #combined_output = weights[0] * torch.cat([pooled_output1, pooled_output2], dim=1) + weights[1] * torch.cat([input_embeddings1, input_embeddings2], dim=1)
        
        logits = self.fc(combined_output)
        ta = self.fc_ta(input_embeddings2)
        te = self.fc_te(input_embeddings1)

        return logits, input_embeddings2, input_embeddings1
        
#######################################################################################################################
# CombinedModelGAT
####################################################################################################################### 
        
class CombinedModelGAT(nn.Module):
    def __init__(self, TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(CombinedModelGAT, self).__init__()

        self.base_models = UniModels(TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout)
        #self.weights = nn.Parameter(torch.randn(2, d_model))
        #self.d_model = d_model + num_cat_var + num_numerical_var
        #self.d_model = d_model + sum(cat_embed_dims) + num_numerical_var
        #self.d_model = sum(cat_embed_dims) + num_numerical_var
        self.d_model = 2 * d_model
        self.dropout = nn.Dropout(dropout)
        #self.fc = torch.nn.Linear(sum(cat_embed_dims) + num_numerical_var, self.d_model)
        #self.bilinear = nn.Bilinear(d_model, sum(cat_embed_dims) + num_numerical_var, d_model)  # Bilinear pooling layer
        
        self.GAT_model = GAT(in_channels=self.d_model, hidden_channels=d_model, out_channels=num_classes)#, heads=4)
            
        self.device = device

    def forward(self, input_ids, attention_mask, categoricals, numericals, edge_index, w):
        #x1, x2 = x[:, :self.d_model], x[:, self.d_model:]
        #x = self.bilinear(x1, x2)
        #x2 = self.fc(x2)
        pooled_output1, pooled_output2, _, _ = self.base_models(input_ids, attention_mask, categoricals, numericals)
        x = torch.cat([pooled_output1, pooled_output2], dim=1)
        x = self.dropout(x)

        logits = self.GAT_model(x, edge_index, w)

        return logits, None, None

#######################################################################################################################
# CrossAttention
#######################################################################################################################

# fix nheads and num_heads
class CrossAttention(nn.Module):
    def __init__(self, TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(CrossAttention, self).__init__()
        self.bert = Bert().to(device)
        #self.bert = DiBert().to(device)
        self.tabular_embed = TabEmbed# TabularEmbedding
        self.CrossAttentionLayer1 = MultiHeadCrossAttention(d_model, d_ff, num_heads=nhead, dropout=ca_dropout).to(device)# d_ff is 768 before hf
        self.CrossAttentionLayer2 = MultiHeadCrossAttention(d_model, d_ff, num_heads=nhead, dropout=ca_dropout).to(device)
        self.bert_self_attention = bert_self_attention

    def forward(self, input_ids, attention_mask, categoricals, numericals):
        # Step 1: Compute tabular embeddings
        tabular_embeddings = self.tabular_embed(categoricals, numericals)  # Shape: (batch_size, num_tab_tokens, embed_dim)

        # Step 2: Get BERT token embeddings 
        #bert_embeddings = self.bert.bert.embeddings(input_ids)  # Shape: (batch_size, seq_len, embed_dim)
        if self.bert_self_attention:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            #last layer embeddings
            bert_embeddings = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, embed_dim) (experient with cls token with each feature [:,0,:].unsqueeze(1) (batch_size, 1, embed_dim))
        else:
            bert_embeddings = self.bert.bert.embeddings(input_ids)  # Shape: (batch_size, seq_len, embed_dim)

        # Weighted combination
        CA1 = self.CrossAttentionLayer1(tabular_embeddings, bert_embeddings)
        CA2 = self.CrossAttentionLayer2(bert_embeddings ,tabular_embeddings)
        
        # pooling
        pooled_output1 = CA1.mean(dim=1)
        pooled_output2 = CA2.mean(dim=1)

        if self.bert_self_attention:
            bert_i_o = bert_embeddings[:,0,:]
        else:
            bert_i_o = bert_embeddings.mean(dim=1)

        return pooled_output1, pooled_output2, tabular_embeddings.mean(dim=1), bert_i_o
        
class SkipBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(SkipBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Identity()
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)

    def forward(self, x, y):
        residual = self.skip(x)
        y = self.linear(y)
        y = self.bn(y)
        y = self.activation(y)
        y = self.dropout(y)
        return y + residual

class FusionSkipNet(nn.Module):
    def __init__(self, TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(FusionSkipNet, self).__init__()

        #self.combModel = UniModels(TabEmbed, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout).to(device)
        self.combModel = UniModels(TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout).to(device)

        # Stack of skip blocks
        layers = []
        hidden_dims = [d_model] * n_layers 
        for dim in hidden_dims:
            layers.append(SkipBlock(d_model, dim, dropout))
            in_dim = dim
        self.skip_layers = nn.ModuleList(layers)

        self.output = nn.Linear(in_dim, num_classes)

    def forward(self, input_ids, attention_mask, categoricals, numericals):
        bert_outputs, tabular_output, _, _ = self.combModel(input_ids, attention_mask, categoricals, numericals)
        #print(bert_outputs.size(), tabular_output.size())
        #x = self.skip_layers(bert_outputs, tabular_output)
        x, y = bert_outputs, tabular_output
        for layer in self.skip_layers:
            y = layer(x, y)
        logits = self.output(y)
        return logits, None, None
    
# class TokenFeatureCrossAttention(nn.Module):
#     def __init__(self, dim_q, dim_kv, dim_out):
#         super().__init__()
#         self.q_proj = nn.Linear(dim_q, dim_out)
#         self.k_proj = nn.Linear(dim_kv, dim_out)
#         self.v_proj = nn.Linear(dim_kv, dim_out)
#         self.attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=4, batch_first=True)

#     def forward(self, q_seq, kv_seq):
#         q_proj = self.q_proj(q_seq)
#         k_proj = self.k_proj(kv_seq)
#         v_proj = self.v_proj(kv_seq)
#         out, _ = self.attn(q_proj, k_proj, v_proj)
#         return out


class GatedResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, original, cross_output):
        residual = self.mlp(cross_output)               # (B, T/F, H)
        gate = self.gate(original)                      # (B, T/F, H)
        fused = original + gate * residual              # Element-wise multiply, same shape
        return self.norm(fused)


# class TokenFeatureCrossAttentionFusionNet(nn.Module):
class CrossAttentionSkipNet(nn.Module):
    def __init__(self, TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout,
                  d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super().__init__()
        self.bert = Bert().to(device)
        self.tabular_embed = TabEmbed# TabularEmbedding
        self.bert_self_attention = bert_self_attention
        
        self.cross_text_to_tab = nn.ModuleList([
            #TokenFeatureCrossAttention(dim_q=hidden_dim, dim_kv=hidden_dim, dim_out=hidden_dim)
            MultiHeadCrossAttention(d_model, d_ff, num_heads=nhead, dropout=ca_dropout).to(device)
            for _ in range(n_layers)
        ])
        self.cross_tab_to_text = nn.ModuleList([
            # TokenFeatureCrossAttention(dim_q=hidden_dim, dim_kv=hidden_dim, dim_out=hidden_dim)
            MultiHeadCrossAttention(d_model, d_ff, num_heads=nhead, dropout=ca_dropout).to(device)
            for _ in range(n_layers)
        ])

        self.res_text = nn.ModuleList([
            GatedResidualBlock(d_ff) for _ in range(n_layers)
        ])
        self.res_tab = nn.ModuleList([
            GatedResidualBlock(d_ff) for _ in range(n_layers)
        ])

        self.text_pool = nn.AdaptiveAvgPool1d(1)
        self.tab_pool = nn.AdaptiveAvgPool1d(1)
        self.Origin_tab_pool = nn.AdaptiveAvgPool1d(1)
        if self.bert_self_attention:
            self.Origin_text_pool = nn.AdaptiveAvgPool1d(1)

        # self.classifier = nn.Sequential(
        #     nn.Linear(2 * hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, num_classes)
        # )

    def forward(self, input_ids, attention_mask, categoricals, numericals):
        B = input_ids.shape[0]

        # Step 1: Compute tabular embeddings
        tabular_embeddings = self.tabular_embed(categoricals, numericals)  # Shape: (batch_size, num_tab_tokens, embed_dim)

        # Step 2: Get BERT token embeddings 
        #bert_embeddings = self.bert.bert.embeddings(input_ids)  # Shape: (batch_size, seq_len, embed_dim)
        if self.bert_self_attention:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            #last layer embeddings
            bert_embeddings = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_len, embed_dim) (experient with cls token with each feature [:,0,:].unsqueeze(1) (batch_size, 1, embed_dim))
        else:
            bert_embeddings = self.bert.bert.embeddings(input_ids)  # Shape: (batch_size, seq_len, embed_dim)

        # 1. Encode text tokens
        text_tokens = bert_embeddings # (B, T, H)
        text_tokens = self.text_proj(text_tokens)  # (B, T, H)

        # 2. Prepare tabular feature embeddings
        tab_tokens = tabular_embeddings                # (B, F, H)

        # 3. Cross-attention with residual updates
        for cross_txt2tab, cross_tab2txt, res_txt, res_tab in zip(
            self.cross_text_to_tab, self.cross_tab_to_text, self.res_text, self.res_tab
        ):
            # Token-level cross-modality attention
            text_to_tab = cross_txt2tab(tab_tokens, text_tokens)  # (B, F, H)
            tab_to_text = cross_tab2txt(text_tokens, tab_tokens)  # (B, T, H)

            # Gated residual updates (shapes are matched internally)
            text_tokens = res_txt(text_tokens, tab_to_text)
            tab_tokens = res_tab(tab_tokens, text_to_tab)

        # 4. Pool each stream
        text_repr = self.text_pool(text_tokens.transpose(1, 2)).squeeze(-1)  # (B, H)
        tab_repr = self.tab_pool(tab_tokens.transpose(1, 2)).squeeze(-1)    # (B, H)

        # Origin representations for tabular and text
        Origin_tab_repr = self.Origin_tab_pool(tabular_embeddings.transpose(1, 2)).squeeze(-1) # (B, H)
        if self.bert_self_attention:
            Origin_text_repr = bert_embeddings[:,0,:]
        else:
            Origin_text_repr = self.Origin_text_pool(bert_embeddings.transpose(1, 2)).squeeze(-1) # (B, H)

        return text_repr, tab_repr, Origin_tab_repr, Origin_text_repr

        # # 5. Final fusion and classification
        # fused = torch.cat([text_repr, tab_repr], dim=1)
        # return self.classifier(fused)

        
#######################################################################################################################
# UniModels
#######################################################################################################################

class UniModels(nn.Module):
    def __init__(self, TabEmbed, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device, bert_self_attention, d_ff, d_fc, ca_dropout):
        super(UniModels, self).__init__()
        
        self.num_cat_var = len(cat_embed_dims)
        # Bert Model
        self.TextModel = OnlyText(self.num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device).to(device)
        self.TabularModel = TabularForBert(self.num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device).to(device)
        #self.fc = nn.Linear(d_model, num_classes)
        

    def forward(self, input_ids, attention_mask, categoricals, numericals):

        # Process text inputs using BERT
        _, bert_outputs, _ = self.TextModel(input_ids, attention_mask, categoricals, numericals)
        _, _, tabular_output = self.TabularModel(input_ids, attention_mask, categoricals, numericals)

        return bert_outputs, tabular_output, None, None
            
#######################################################################################################################
# BertWithTabular
#######################################################################################################################       
class BertWithTabular(nn.Module):
    def __init__(self, num_cat_var, num_numerical_var, nhead, n_layers, cat_embed_dims, num_classes, dropout, d_model, device):
        """
        BertWithTabular integrates textual and tabular data by extending BERT's input space with tabular embeddings.
        
        Args:
            num_cat_var (int): Number of categorical variables.
            num_numerical_var (int): Number of numerical variables.
            nhead (int): Not used in this model (reserved for future extensions).
            n_layers (int): Not used in this model (reserved for future extensions).
            cat_embed_dims (list): List of embedding dimensions for each categorical variable.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
            d_model (int): Dimensionality of embeddings (should match BERT's hidden size).
            device (str or torch.device): Device to run the model on.
        """
        super(BertWithTabular, self).__init__()
        self.bert = Bert().to(device)
        self.tabular_embed = TabularEmbedding(cat_embed_dims, num_numerical_var, d_model)
        self.embed_dim = d_model
        #self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.fc2 = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask, categoricals, numericals):
        """
        Forward pass for BertWithTabular.
        
        Args:
            input_ids (torch.Tensor): Tokenized input IDs for text (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask for text tokens (batch_size, seq_len).
            categoricals (torch.Tensor): Categorical input data (batch_size, num_cat_vars).
            numericals (torch.Tensor): Numerical input data (batch_size, num_num_vars).
        
        Returns:
            torch.Tensor: Logits for classification (batch_size, num_classes).
        """
        # Compute tabular embeddings
        tabular_embeddings = self.tabular_embed(categoricals, numericals)  # Shape: (batch_size, num_tab_tokens, embed_dim)
        tabular_token_count = tabular_embeddings.size(1)

        # Get BERT token embeddings
        bert_embeddings = self.bert.bert.embeddings(input_ids)  # Shape: (batch_size, seq_len, embed_dim)

        # Concatenate BERT and tabular embeddings
        combined_embeddings = torch.cat([bert_embeddings, tabular_embeddings], dim=1)  # Shape: (batch_size, seq_len + num_tab_tokens, embed_dim)

        # Extend attention mask
        batch_size, seq_len = input_ids.shape
        tabular_attention_mask = torch.ones((batch_size, tabular_token_count), dtype=torch.long, device=attention_mask.device)
        extended_attention_mask = torch.cat([attention_mask, tabular_attention_mask], dim=1)  # Shape: (batch_size, seq_len + num_tab_tokens)

        # Pass through BERT encoder
        extended_attention_mask = self.bert.bert.get_extended_attention_mask(extended_attention_mask, combined_embeddings.shape[:2], device=input_ids.device)
        encoder_outputs = self.bert.bert.encoder(hidden_states=combined_embeddings, attention_mask=extended_attention_mask)[0]

        # Extract [CLS] token representation and classify
        cls_output = encoder_outputs[:, 0, :]  # Shape: (batch_size, embed_dim)
        #logits = self.fc(cls_output)  # Shape: (batch_size, num_classes)
        logits = self.fc2(cls_output)
        return logits, None, None     

#########################################################################################################
#########################################################################################################
#Helping Models
#########################################################################################################
#########################################################################################################

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

#BertClassifier3
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze all BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze the last layer
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
        
class DiBert(nn.Module):
    def __init__(self):
        super(DiBert, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
class TabularEmbedding(nn.Module):
    def __init__(self, cat_vocab_sizes, num_numerical_vars, embed_dim, 
                 numerical_transform='linear', transform_params=None):
        """
        Unified TabularEmbedding class supporting multiple numerical transformation methods.
        
        Args:
            cat_vocab_sizes (list): List of sizes of categorical vocabularies.
            num_numerical_vars (int): Number of numerical variables.
            embed_dim (int): Embedding dimension for categorical variables and numerical transformation.
            numerical_transform (str): Type of numerical transformation. Options:
                - 'linear': Simple linear transformation
                - 'fourier': Fourier feature expansion
                - 'fourier_vec': Fourier features with vectorized output
                - 'rbf': Radial Basis Function expansion
                - 'rbf_vec': RBF with vectorized output
                - 'chebyshev': Chebyshev polynomial expansion
                - 'sigmoid': Sigmoid basis functions
                - 'positional': Sinusoidal positional encoding
            transform_params (dict): Parameters specific to the transformation method:
                - For 'fourier'/'fourier_vec': {'num_fourier_terms': int}
                - For 'rbf'/'rbf_vec': {'num_rbf_centers': int, 'rbf_sigma': float}
                - For 'chebyshev': {'num_chebyshev_terms': int}
                - For 'sigmoid': {'num_sigmoid_basis': int}
        """
        super(TabularEmbedding, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_numerical_vars = num_numerical_vars
        self.numerical_transform = numerical_transform
        self.transform_params = transform_params or {}
        
        # Embedding layers for categorical variables
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in cat_vocab_sizes
        ])
        
        # Initialize numerical transformers based on the selected method
        self._init_numerical_transformers()
        
    def _init_numerical_transformers(self):
        """Initialize numerical transformers based on the selected transformation method."""
        if self.numerical_transform == 'linear':
            self.num_transformers = nn.ModuleList([
                nn.Linear(1, self.embed_dim) for _ in range(self.num_numerical_vars)
            ])
            
        elif self.numerical_transform == 'fourier':
            self.num_fourier_terms = self.transform_params.get('num_fourier_terms', 5)
            self.num_transformers = nn.ModuleList([
                nn.Linear(2 * self.num_fourier_terms + 1, self.embed_dim) 
                for _ in range(self.num_numerical_vars)
            ])
            
        elif self.numerical_transform == 'fourier_vec':
            self.num_fourier_terms = self.embed_dim
            self.num_transformers = nn.ModuleList([
                nn.Linear(1, self.embed_dim) for _ in range(self.num_numerical_vars)
            ])
            
        elif self.numerical_transform == 'rbf':
            self.num_rbf_centers = self.transform_params.get('num_rbf_centers', 5)
            self.rbf_sigma = self.transform_params.get('rbf_sigma', 1.0)
            self.rbf_centers = nn.Parameter(torch.linspace(-1, 1, self.num_rbf_centers).view(1, -1))
            self.num_transformers = nn.ModuleList([
                nn.Linear(self.num_rbf_centers, self.embed_dim) 
                for _ in range(self.num_numerical_vars)
            ])
            
        elif self.numerical_transform == 'rbf_vec':
            self.num_rbf_centers = self.embed_dim
            self.rbf_sigma = self.transform_params.get('rbf_sigma', 1.0)
            self.rbf_centers = nn.Parameter(torch.linspace(-1, 1, self.num_rbf_centers).view(1, -1))
            self.num_transformers = nn.ModuleList([
                nn.Linear(self.num_rbf_centers, self.embed_dim) 
                for _ in range(self.num_numerical_vars)
            ])
            
        elif self.numerical_transform == 'chebyshev':
            self.num_chebyshev_terms = self.transform_params.get('num_chebyshev_terms', 3)
            self.num_transformers = nn.ModuleList([
                nn.Linear(self.num_chebyshev_terms, self.embed_dim) 
                for _ in range(self.num_numerical_vars)
            ])
            
        elif self.numerical_transform == 'sigmoid':
            self.num_sigmoid_basis = self.transform_params.get('num_sigmoid_basis', 5)
            self.sigmoid_centers = nn.Parameter(torch.linspace(-1, 1, self.num_sigmoid_basis))
            self.sigmoid_scales = nn.Parameter(torch.ones(self.num_sigmoid_basis))
            self.num_transformers = nn.ModuleList([
                nn.Linear(self.num_sigmoid_basis, self.embed_dim) 
                for _ in range(self.num_numerical_vars)
            ])
            
        elif self.numerical_transform == 'positional':
            self.num_transformers = nn.ModuleList([
                nn.Linear(self.embed_dim, self.embed_dim) 
                for _ in range(self.num_numerical_vars)
            ])
            
        else:
            raise ValueError(f"Unknown numerical_transform: {self.numerical_transform}")
    
    def _apply_numerical_transform(self, x, var_idx):
        """Apply the selected numerical transformation to input x for variable var_idx."""
        if self.numerical_transform == 'linear':
            return self.num_transformers[var_idx](x)
            
        elif self.numerical_transform == 'fourier':
            fourier_features = self._fourier_transform(x)
            return self.num_transformers[var_idx](fourier_features)
            
        elif self.numerical_transform == 'fourier_vec':
            fourier_features = self._fourier_transform_vec(x)
            return fourier_features * self.num_transformers[var_idx](x)
            
        elif self.numerical_transform == 'rbf':
            rbf_features = self._rbf_transform(x)
            return self.num_transformers[var_idx](rbf_features)
            
        elif self.numerical_transform == 'rbf_vec':
            rbf_features = self._rbf_transform_vec(x)
            return rbf_features * self.num_transformers[var_idx](x)
            
        elif self.numerical_transform == 'chebyshev':
            chebyshev_features = self._chebyshev_transform(x)
            return self.num_transformers[var_idx](chebyshev_features)
            
        elif self.numerical_transform == 'sigmoid':
            sigmoid_features = self._sigmoid_basis_transform(x)
            return self.num_transformers[var_idx](sigmoid_features)
            
        elif self.numerical_transform == 'positional':
            pos_encoded = self._positional_encoding(x)
            return self.num_transformers[var_idx](pos_encoded)
    
    def _fourier_transform(self, x):
        """Fourier feature expansion."""
        features = [torch.ones_like(x)]
        for k in range(1, self.num_fourier_terms + 1):
            features.append(torch.sin(k * x))
            features.append(torch.cos(k * x))
        return torch.cat(features, dim=-1)
    
    def _fourier_transform_vec(self, x):
        """Vectorized Fourier feature expansion."""
        features = [torch.ones_like(x)]
        for k in range(1, self.num_fourier_terms//2 + 1):
            features.append(torch.sin(k * x))
            features.append(torch.cos(k * x))
        features.append(torch.sin((k+1) * x))
        return torch.cat(features, dim=-1)
    
    def _rbf_transform(self, x):
        """RBF expansion."""
        x = x.unsqueeze(-1)
        rbf_features = torch.exp(-((x - self.rbf_centers) ** 2) / (2 * self.rbf_sigma ** 2))
        return rbf_features.squeeze(1)  # Remove the extra dimension
    
    def _rbf_transform_vec(self, x):
        """Vectorized RBF expansion."""
        x = x.unsqueeze(-1)
        rbf_features = torch.exp(-((x - self.rbf_centers) ** 2) / (2 * self.rbf_sigma ** 2))
        return rbf_features.squeeze(1)  # Remove the extra dimension
    
    def _chebyshev_transform(self, x):
        """Chebyshev polynomial expansion."""
        x = x.unsqueeze(-1)
        chebyshev_features = [torch.ones_like(x), x]
        
        for k in range(2, self.num_chebyshev_terms):
            chebyshev_features.append(2 * x * chebyshev_features[-1] - chebyshev_features[-2])
        
        return torch.cat(chebyshev_features, dim=-1).squeeze(1)
    
    def _sigmoid_basis_transform(self, x):
        """Sigmoid basis function expansion."""
        x = x.unsqueeze(-1)
        sigmoid_features = torch.sigmoid(self.sigmoid_scales * (x - self.sigmoid_centers))
        return sigmoid_features.squeeze(1)
    
    def _positional_encoding(self, x):
        """Sinusoidal positional encoding."""
        batch_size = x.shape[0]
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=x.device) * 
                           (-torch.log(torch.tensor(10000.0)) / self.embed_dim))
        x_expanded = x.expand(batch_size, self.embed_dim // 2)
        
        pe = torch.zeros(batch_size, self.embed_dim, device=x.device)
        pe[:, 0::2] = torch.sin(x_expanded * div_term)
        pe[:, 1::2] = torch.cos(x_expanded * div_term)
        
        return pe
    
    def forward(self, categoricals, numericals):
        """
        Args:
            categoricals (torch.Tensor): Categorical input (batch_size, num_cat_vars).
            numericals (torch.Tensor): Numerical input (batch_size, num_num_vars).
        
        Returns:
            torch.Tensor: Combined tabular embeddings (batch_size, total_tabular_tokens, embed_dim).
        """
        numericals = numericals.float()
        
        # Embed categorical variables
        cat_embeddings = [emb(categoricals[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embeddings = torch.stack(cat_embeddings, dim=1)
        
        # Transform numerical variables
        num_embeddings = []
        for i in range(self.num_numerical_vars):
            x = numericals[:, i:i+1]
            transformed = self._apply_numerical_transform(x, i)
            num_embeddings.append(transformed)
        
        num_embeddings = torch.stack(num_embeddings, dim=1)
        
        # Concatenate embeddings along the token axis
        combined_embeddings = torch.cat([cat_embeddings, num_embeddings], dim=1)
        
        return combined_embeddings


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.5):#dropout=0.5 for not self attention otherwise 0.1
        super(MultiHeadCrossAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # Size per head
        
        # Linear layers for query, key, and value projections
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # Layer normalizations
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.value_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, input_dim)# nn.Linear(hidden_dim, hidden_dim)
        
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        """
        x1: (batch_size, seq_len_1, input_dim) -> Query
        x2: (batch_size, seq_len_2, input_dim) -> Key & Value
        """
        batch_size, seq_len_1, _ = x1.shape
        _, seq_len_2, _ = x2.shape

        # Project inputs to query, key, and value
        query = self.query_norm(self.query(x1))  # (B, S_1, H)
        key = self.key_norm(self.key(x2))        # (B, S_2, H)
        value = self.value_norm(self.value(x2))  # (B, S_2, H)

        # Reshape to multiple heads: (B, num_heads, S, head_dim)
        query = query.view(batch_size, seq_len_1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len_2, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len_2, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Compute attention output
        attended_output = torch.matmul(attention_weights, value)  # (B, num_heads, S_1, head_dim)

        # Merge heads back: (B, S_1, H)
        attended_output = attended_output.transpose(1, 2).contiguous().view(batch_size, seq_len_1, -1)

        # Final linear projection
        attended_output = self.out_proj(attended_output)

        return attended_output
        
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=1)
    
    def forward(self, x, edge_index, w):
        #x = self.linear(x)
        x = F.elu(self.conv1(x, edge_index, w))
        x = self.conv2(x, edge_index, w)
        return x#F.log_softmax(x, dim=1)
        
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        # GCN layers
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        #x = self.linear(x)  # Apply linear transformation
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x#F.log_softmax(x, dim=1)

#########################################################################################################
#########################################################################################################
#Helping Models
#########################################################################################################
######################################################################################################### 

def init_model(model_type, d_model, max_len, vocab_size, cat_vocab_sizes, 
               num_cat_var, num_numerical_var, quantiles, n_heads,
               d_ff, n_layers, dropout, d_fc, n_classes, seed, device, text_model="", ca_dropout=0.1):
    
    TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='linear')
    bert_self_attention = False
    if "Fourier" in model_type:
        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='fourier')#d_ff
    elif "PosEnVec" in model_type:
        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='positional')
    elif "FourierVec" in model_type:
        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='fourier_vec')
    elif "RBF" in model_type:
        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='rbf')
    elif "RBFVec" in model_type:
        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='rbf_vec')
    elif "Sigmoid" in model_type:
        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='sigmoid')
    elif "Chebyshev" in model_type:    
        TabEmbed = TabularEmbedding(cat_vocab_sizes, num_numerical_var, d_model, numerical_transform='chebyshev')
    
    if "CombinedModel" in model_type:
        MultiModelsObj = UniModels
    elif "CrossAttention" in model_type:
        MultiModelsObj = CrossAttention
    elif "CrossAttentionSkipNet" in model_type:
        MultiModelsObj = CrossAttentionSkipNet

    if model_type in ["TTT-SRP", "TTT-PCA", "TTT-Kaiming"]:
        model_type = "TTT"
    if model_type == "TTT":
        torch.manual_seed(seed)
        model =  TTT(d_model, 
                              max_len, 
                              vocab_size,
                              cat_vocab_sizes, 
                              num_cat_var,
                              num_numerical_var,
                              quantiles,
                              n_heads,
                              d_ff, 
                              n_layers, 
                              dropout,
                              d_fc,
                              n_classes,
                              device)
                        
    if model_type == "LateFuseBERT":
        torch.manual_seed(seed)
        model = LateFuseBERT(text_model,
                             cat_vocab_sizes,
                             num_cat_var,
                             num_numerical_var,
                             d_model,
                             n_heads,
                             n_layers, 
                             dropout,
                             d_fc,
                             n_classes)
                                 
    if model_type == "AllTextBERT":
        torch.manual_seed(seed)
        model = AllTextBERT(text_model,
                            d_model,
                            dropout,
                            d_fc,
                            n_classes)

    if model_type == "TabularForBert":
        torch.manual_seed(seed)
        model = TabularForBert(num_cat_var, 
                                num_numerical_var, 
                                n_heads, 
                                n_layers, 
                                cat_vocab_sizes, 
                                n_classes, 
                                dropout, 
                                d_model, 
                                device)
    
    if model_type == "OnlyTabular":
        torch.manual_seed(seed)
        model = OnlyTabular(num_cat_var, 
                                num_numerical_var, 
                                n_heads, 
                                n_layers, 
                                cat_vocab_sizes, 
                                n_classes, 
                                dropout, 
                                device)
        
    if model_type == "OnlyText":
        torch.manual_seed(seed)
        model = OnlyText(num_cat_var, 
                         num_numerical_var, 
                         n_heads, 
                         n_layers, 
                         cat_vocab_sizes, 
                         n_classes, 
                         dropout, 
                         d_model, 
                         device)
                         
    if model_type == "CombinedModelGAT":
        torch.manual_seed(seed)
        model = CombinedModelGAT(TabEmbed,
                         num_cat_var, 
                         num_numerical_var, 
                         n_heads, 
                         n_layers, 
                         cat_vocab_sizes, 
                         n_classes, 
                         dropout, 
                         d_model, 
                         device,
                         bert_self_attention,
                         d_ff,
                         d_fc,
                         ca_dropout)
    if model_type == "FusionSkipNet":
        torch.manual_seed(seed)
        model = FusionSkipNet(TabEmbed,
                         num_cat_var, 
                         num_numerical_var, 
                         n_heads, 
                         n_layers, 
                         cat_vocab_sizes, 
                         n_classes, 
                         dropout, 
                         d_model, 
                         device,
                         bert_self_attention,
                         d_ff,
                         d_fc,
                         ca_dropout)

        
    if model_type[-5:] == "SumW2":
        torch.manual_seed(seed)
        if "SumW2s" in model_type:
            bert_self_attention = True
        model = CombinedModelSumW2(MultiModelsObj,
                                    TabEmbed,
                                    num_cat_var, 
                                    num_numerical_var, 
                                    n_heads, 
                                    n_layers, 
                                    cat_vocab_sizes, 
                                    n_classes, 
                                    dropout, 
                                    d_model, 
                                    device,
                                    bert_self_attention,
                                    d_ff,
                                    d_fc,
                                    ca_dropout)
        
    if "Concat2" in model_type:
        torch.manual_seed(seed)
        if "Concat2s" in model_type:
            bert_self_attention = True
        model = CombinedModelConcat2(MultiModelsObj,
                                    TabEmbed,
                                    num_cat_var, 
                                    num_numerical_var, 
                                    n_heads, 
                                    n_layers, 
                                    cat_vocab_sizes, 
                                    n_classes, 
                                    dropout, 
                                    d_model, 
                                    device,
                                    bert_self_attention,
                                    d_ff,
                                    d_fc,
                                    ca_dropout)
                                    
    if "SumW4" in model_type:
        torch.manual_seed(seed)
        if "SumW4s" in model_type:
            bert_self_attention = True
        model = CombinedModelSumW4(MultiModelsObj,
                                    TabEmbed,
                                    num_cat_var, 
                                    num_numerical_var, 
                                    n_heads, 
                                    n_layers, 
                                    cat_vocab_sizes, 
                                    n_classes, 
                                    dropout, 
                                    d_model, 
                                    device,
                                    bert_self_attention,
                                    d_ff,
                                    d_fc,
                                    ca_dropout)
    
    if "Concat4" in model_type:
        torch.manual_seed(seed)
        if "Concat4s" in model_type:
            bert_self_attention = True
        model = CombinedModelConcat4(MultiModelsObj,
                                    TabEmbed,
                                    num_cat_var, 
                                    num_numerical_var, 
                                    n_heads, 
                                    n_layers, 
                                    cat_vocab_sizes, 
                                    n_classes, 
                                    dropout, 
                                    d_model, 
                                    device,
                                    bert_self_attention,
                                    d_ff,
                                    d_fc,
                                    ca_dropout)
    if model_type[-6:] == "Dgate1":
        torch.manual_seed(seed)
        if "Dgate1s" in model_type:
            bert_self_attention = True
        model = CombinedModelDgate1(MultiModelsObj,
                                    TabEmbed,
                                    num_cat_var, 
                                    num_numerical_var, 
                                    n_heads, 
                                    n_layers, 
                                    cat_vocab_sizes, 
                                    n_classes, 
                                    dropout, 
                                    d_model, 
                                    device,
                                    bert_self_attention,
                                    d_ff,
                                    d_fc,
                                    ca_dropout)
    if model_type == "BertWithTabular":
        torch.manual_seed(seed)
        model = BertWithTabular(num_cat_var,
                             num_numerical_var,
                             n_heads,
                             n_layers,
                             cat_vocab_sizes, 
                             n_classes,
                             dropout,
                             d_model,
                             device)
    
    return model
