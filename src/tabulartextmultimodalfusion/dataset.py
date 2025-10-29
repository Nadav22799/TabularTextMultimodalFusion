import torch
#import torchtext
#from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, TensorDataset
import re
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
import pandas as pd
from collections import Counter
from settings import * # settings

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

import random
from collections import defaultdict

class TripletDataset(Dataset):
    def __init__(self, data_loader):
        self.data = []  # To store x = (text, mask, categorical, numerical)
        self.labels = []  # To store y

        # Collect all data from the provided data loader
        for batch in data_loader:
            text, categorical, numerical, y, mask = batch
            x = (text, categorical, numerical, mask)
            self.data.extend(list(zip(*x)))  # Flatten x into tuples and store
            self.labels.extend(y)

        # Group samples by label (0 to n)
        self.class_dict = defaultdict(list)
        for idx, label in enumerate(self.labels):
            label = label.item()  # Convert to integer
            self.class_dict[label].append(self.data[idx])

    def __len__(self):
        return len(self.labels)

    def get_random_sample(self, samples):
        return random.choice(samples)

    def __getitem__(self, index):
        # Anchor selection
        anchor = self.data[index]
        anchor_label = self.labels[index].item()

        # Positive sample: same label, different instance
        positive = self.get_random_sample(self.class_dict[anchor_label])
        while torch.equal(positive[0], anchor[0]):
            positive = self.get_random_sample(self.class_dict[anchor_label])

        # Negative sample: different class
        negative_label = random.choice(list(self.class_dict.keys()))
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_dict.keys()))

        negative = self.get_random_sample(self.class_dict[negative_label])

        # Unpack x from tuple format
        anchor = (anchor[0], anchor[1], anchor[2], anchor[3])  # (text, categorical, numerical, mask)
        positive = (positive[0], positive[1], positive[2], positive[3])
        negative = (negative[0], negative[1], negative[2], negative[3])

        # Return triplets with labels
        return (anchor, anchor_label), (positive, anchor_label), (negative, negative_label)

# load and prepare dataset
def preprocess_dataset(dataset, model_type):

    # load parameters
    FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT= load_settings(dataset)

    # cloth
    if dataset == "cloth":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # rename label
        df["rating"] = df["Rating"].copy() 
        df = df.rename({"Rating":"Y"}, axis=1) 
        df['Y'] = df['Y'] - 1 # starts from 0
        df.loc[df.Title.isnull(),'Title'] = '' # replace NaN title with ''
        df.loc[df['Review Text'].isnull(),'Title'] = '' # drop NaN reviews (as title is too short)
        # concatenate title and review text
        df[text_var] = df['Title'] + ' ' + df['Review Text']
        # drop na
        df = df.dropna().reset_index()
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y']]
    
    # wine
    if dataset == "wine_10" or dataset == "wine_100":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # rename label
        df["Variety"] = df["variety"].copy()
        df = df.rename({"variety":"Y"}, axis=1)
        # extract year from title
        yearSearch = []
        for value in df['title']:
            regexresult = re.search(r'19\d{2}|20\d{2}', value)
            if regexresult:
                yearSearch.append(regexresult.group())
            else:
                yearSearch.append(None)
        df['year'] = yearSearch
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y', 'Variety']]
        # drop na
        df = df.dropna().reset_index(drop=True)
        # label Y: keep only most frequent labels
        df = df[df.Y.isin(df['Y'].value_counts(ascending=False).index[:N_CLASSES].tolist())]
        # label encoding of target variable
        le = LabelEncoder()
        df['Y'] = le.fit_transform(df['Y'])
        

    # kick
    if dataset == "kick":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # rename label
        df = df.rename({"final_status":"Y"}, axis=1) 
        # add duration to launch (in days)
        df['duration'] = (df['deadline'] - df['launched_at'])/(3600*24)
        # log transformation for goal
        df['log_goal'] = np.log10(df['goal'])
        # concatenate name and desc
        df[text_var] = df['name'] + ' ' + df['desc']
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y']]
        # drop na
        df = df.dropna().reset_index(drop=True)
        # format
        df['disable_communication'] = df['disable_communication'].astype(str)
        
        
    # airbnb
    if dataset == "airbnb":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # target binning
        bin_edges = np.quantile(df['price'], q = np.arange(N_CLASSES+1)/10)
        bin_edges[0] = 0 # start at 0
        labels = np.arange(N_CLASSES)
        df['Y'] = pd.cut(df['price'], bins = bin_edges, labels = labels)
        # concatenate text fields
        df = df[~((df.summary.isnull()) & (df.description.isnull()))] # drop rows where both fields are empty
        df.loc[df.name.isnull(),'name']= '' # replace NaN name with ''
        df.loc[df.summary.isnull(),'summary']= '' # replace NaN summary with ''
        df.loc[df.description.isnull(),'description']= '' # replace NaN description with ''
        df[text_var] = df['name'] + ' ' + df['summary']+ ' ' + df['description']
        # feature extraction
        df['host_since_year'] = df['host_since'].str.extract('.*(\d{4})', expand = False)
        df['last_review_year'] = df['last_review'].str.extract('.*(\d{4})', expand = False)
        df['host_response_rate'] = df['host_response_rate'].str.replace('%','')
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y']]  
        # drop na
        df = df.dropna().reset_index(drop=True)
        # format
        df['host_response_rate'] = df['host_response_rate'].astype(int)
        
    # petfinder
    if dataset == "pet":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # rename label
        df = df.rename({"AdoptionSpeed":"Y"}, axis=1) 
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y']]
        # drop na
        df = df.dropna().reset_index()
        # format
        df[categorical_var] = df[categorical_var].astype(str) 

    
    # salary    
    if dataset == "salary":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # rename label
        df = df.rename({"salary":"Y"}, axis=1) 
        # compute years of experience
        df['experience_int'] = df['experience'].str.split("-").str.get(0)
        # concatenate text fields
        df.loc[df.job_description.isnull(),'job_description']= '' # replace NaN job_description with ''
        df.loc[df.job_desig.isnull(),'job_desig']= '' # replace NaN job_desig with ''
        df.loc[df.key_skills.isnull(),'key_skills']= '' # replace NaN key_skills with ''
        df[text_var] = df['job_description'] + ' ' + df['job_desig']+ ' ' + df['key_skills']
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y']]  
        # drop na
        df = df.dropna().reset_index(drop=True)
        # format
        df[categorical_var] = df[categorical_var].astype(str) 
        df[numerical_var] = df[numerical_var].astype(int) 
        # label encoding of target variable
        le = LabelEncoder()
        df['Y'] = le.fit_transform(df['Y'])
        
    # jigsaw
    if dataset == "jigsaw":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # target definition
        df["Y"] = np.where(df["target"]>=0.5,1,0)
        # drop na
        df = df.dropna().reset_index()
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y']]
        
    if dataset == "income":
        # load data
        df = pd.read_csv("datasets/" + FILENAME)
        # rename label
        df["income"] = df["income"].copy() 
        df = df.rename({"income":"Y"}, axis=1) 
        df['Y'] = df['Y'].replace({'<=50K': 0, '>50K': 1}) # 
        #df.loc[df.Title.isnull(),'Title'] = '' # replace NaN title with ''
        #df.loc[df['Review Text'].isnull(),'Title'] = '' # drop NaN reviews (as title is too short)
        # concatenate all text columns
        #df[text_var] = df['Title'] + ' ' + df['Review Text']
        text_cols = ['workclass','education','native.country', 'relationship', 'occupation', 'marital.status']
        text_list = []
        for i in range(len(df)):
          text = ""
          for feature, value in zip(df[text_cols].iloc[i].index, df[text_cols].iloc[i].values):
            text += str(feature) + " " + str(value) + " "
          text_list.append(text)
        df[text_var] = text_list
        # drop na
        df = df.dropna().reset_index()
        # drop unused columns
        df = df[categorical_var + numerical_var + [text_var, 'Y']]

    # for AllTextBERT, we concatenate the text and tabular variables
    if model_type == "AllTextBERT":
        text_list = []
        for i in range(len(df)):
          text = ""
          for feature, value in zip(df[categorical_var+numerical_var].iloc[i].index, df[categorical_var+numerical_var].iloc[i].values):
            text += str(feature) + " " + str(value) + " "
          text_list.append(text)
        df[text_var] = (text_list + df[text_var].values ) 
        
    return df

# text functions

def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text) # keep only words/numbers and spaces
    return text

#def tokenize (text):
#    text = re.sub(r'[0-9]', '', text) # remove numbers
#    text = re.sub(r'[^\w\s]', ' ', text) # keep only words and space
#    tokenizer = get_tokenizer("basic_english") # normalize and tokenize
#    tokens = tokenizer(text)
#    return tokens
    
    
def tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[0-9]', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # Keep only words and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = text.split()  # Split into tokens by spaces
    return tokens

def encode_sentence(text, vocab2index, max_len):
    tokenized = tokenize(text)
    encoded = np.zeros(max_len, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(max_len, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded 

def vocabulary(df, text_field):
            
    # Count number of occurences of each word
    counts = Counter()
    for index, row in df.iterrows():
        counts.update(tokenize(row[text_field]))

    # deleting infrequent words
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    # Create vocabulary
    vocab2index = {"":0, "UNK":1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    vocab_size = len(words) 

    return vocab2index, vocab_size, words


# categorical and numerical variable functions

def ordinalEncoding(source, target, var_list):
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    reviews_cat_encoded = oe.fit_transform(source[var_list])
    target_cat_encoded = oe.transform(target[var_list])
    categorical_var_oe = []
           
    for idx, var in enumerate(var_list):
        source[var+' - oe'] = reviews_cat_encoded[:,idx].astype(int) + 1
        target[var+' - oe'] = target_cat_encoded[:,idx].astype(int) + 1 # add 1 so that unknown token is 0
        categorical_var_oe.append(var+' - oe')
    cat_vocab_sizes = [len(categ_list) + 1 for categ_list in oe.categories_] # +1 for unknown token
    return categorical_var_oe, cat_vocab_sizes

def standardScaling(source, target, var_list):
    numerical_var_scaled = [var + " - scaled" for var in var_list]
    sc = StandardScaler()
    source[numerical_var_scaled] = pd.DataFrame(sc.fit_transform(source[var_list]), columns = numerical_var_scaled).values
    target[numerical_var_scaled] = pd.DataFrame(sc.transform(target[var_list]), columns = numerical_var_scaled).values
    return numerical_var_scaled
 
def oneHotEncoding(source, target, var_list):
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe_array = ohe.fit_transform(source[var_list].values).toarray()
    ohe_var = ohe.get_feature_names_out().tolist()
    source[ohe_var] = ohe_array
    ohe_array = ohe.transform(target[var_list].values).toarray()
    target[ohe_var] = ohe_array

    return ohe_var
    
    
# Custom dataset

class CustomDataset(Dataset):
    def __init__(self, texts, categoricals, numericals, labels):
        # text
        self.texts = texts
        # categorical variables
        self.categoricals = categoricals
        # tabulars (continuous)
        self.numericals = numericals
        # label
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        text = self.texts[idx]
        categorical = self.categoricals.iloc[idx].values
        numerical = self.numericals.iloc[idx].values
        label = self.labels[idx]
            
        return text, categorical, numerical, label
    
    
def prepareCustomDatasets(df_train, df_validation, target, encoded_text_var, categorical_var, numerical_var, label):

    # training dataset and loader
    dataset_train = CustomDataset(
                            texts = df_train[encoded_text_var].values,
                            categoricals = df_train[categorical_var],
                            numericals = df_train[numerical_var],
                            labels = df_train[label].values)


    # validation dataset and loader
    dataset_validation = CustomDataset(
                            texts = df_validation[encoded_text_var].values,
                            categoricals = df_validation[categorical_var],
                            numericals = df_validation[numerical_var],
                            labels = df_validation[label].values)


    # Target dataset and loader
    dataset_target = CustomDataset(
                            texts = target[encoded_text_var].values,
                            categoricals = target[categorical_var],
                            numericals = target[numerical_var],
                            labels = target[label].values)
    
    return dataset_train, dataset_validation, dataset_target
    
def prepareDFDatasetForXGBoost(df, categorical_var, numerical_var, label):


    # Combine categorical and numerical features
    X = np.hstack([
        df[categorical_var].values,
        df[numerical_var].values
    ]).astype(np.float32)  # Ensure float for XGBoost compatibility

    y = df[label].values
    return X,y
    
    
def prepareTensorDatasetWithTokenizer(df, clean_text, categorical_var, numerical_var, label, tokenizer, max_len, special_tokens, model_type):
    
    texts = df[clean_text].values
    categoricals = df[categorical_var].values
    numericals = df[numerical_var].values
    labels = df[label].values

    # tokenize dataset
    input_ids = []
    attention_masks = []
    for sent in texts:
        # Tokenize the text and add '[CLS]' and '[SEP]'
        encoded_dict = tokenizer.encode_plus(sent,# Sentence to encode.
                                          add_special_tokens = special_tokens, # Add '[CLS]' and '[SEP]'
                                          max_length = max_len, # Pad & truncate all sentences
                                          pad_to_max_length = True,
                                          truncation=True,
                                          return_attention_mask = True, # Construct attn. masks
                                          return_tensors = 'pt',
                                          )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    categoricals = torch.tensor(categoricals)
    numericals = torch.tensor(numericals)
    labels = torch.tensor(labels)
    
    # Combine the inputs into a TensorDataset.
    if model_type in ["TTT-SRP", "TTT-PCA", "TTT-Kaiming"]:
        dataset = TensorDataset(input_ids, categoricals, numericals, labels) # attention mask is computed separately (see optimization step)
    else:
        dataset = TensorDataset(input_ids, categoricals, numericals, labels, attention_masks)
    
    return dataset
    
def encode_text_numeric(input_ids, attention_masks, categoricals, numericals, model, tokenizer, cat_vocab_sizes, device="cpu"):
    """ Converts a list of texts into BERT embeddings """
    model.to(device)
    model.eval()
    
    encoded_np_numeric = []
    encoded_np_categoricals = []
    encoded_np_tabular = []
    encoded_np_text = []
    encoded_np_all = []
    with torch.no_grad():
        for input_ids_i, attention_masks_i, categoricals_i, numericals_i in zip(input_ids, attention_masks, categoricals, numericals):
            input_ids_i = input_ids_i.unsqueeze(0).to(device)
            attention_masks_i = attention_masks_i.unsqueeze(0).to(device)
            categoricals_i = categoricals_i.unsqueeze(0).to(device)
            numericals_i = numericals_i.unsqueeze(0).to(device)
            outputs = model(input_ids_i, attention_masks_i)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation
            
            # Generate one-hot encoding for each categorical variable
            one_hot_encoded = [F.one_hot(categoricals_i[:, i], num_classes=cat_vocab_sizes[i]) for i in range(len(cat_vocab_sizes))]
            
            # Concatenate the one-hot vectors along the last dimension, or create empty tensor if none
            if len(one_hot_encoded) > 0:
                one_hot_encoded = torch.cat(one_hot_encoded, dim=-1)
            else:
                batch_size = numericals_i.shape[0]
                one_hot_encoded = torch.empty(batch_size, 0, device=numericals_i.device)
            categoricals_i = one_hot_encoded
            
            #numeric_embedding = torch.cat([categoricals_i.float(), numericals_i.float()], dim=1)
            numeric_embedding = numericals_i.float()
            categoricals_embedding = categoricals_i.float()
            all_embedding = torch.cat([cls_embedding, categoricals_i.float(), numericals_i.float()], dim=1)
            tabular_embedding = torch.cat([categoricals_i.float(), numericals_i.float()], dim=1)
            numeric_embedding = numeric_embedding.squeeze()
            categoricals_embedding = categoricals_embedding.squeeze()
            cls_embedding = cls_embedding.squeeze()
            tabular_embedding = tabular_embedding.squeeze()
            all_embedding = all_embedding.squeeze()
            encoded_np_numeric.append(numeric_embedding.cpu().numpy())
            encoded_np_categoricals.append(categoricals_embedding.cpu().numpy())
            encoded_np_tabular.append(tabular_embedding.cpu().numpy())
            encoded_np_text.append(cls_embedding.cpu().numpy())
            encoded_np_all.append(all_embedding.cpu().numpy())
    
    return np.array(encoded_np_all),  np.array(encoded_np_text), np.array(encoded_np_tabular), np.array(encoded_np_numeric), np.array(encoded_np_categoricals)  # Shape: (N, D)

def build_knn_graph(features, k=5, metric='euclidean'):
    """ Constructs a graph using KNN """
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
    _, indices = nbrs.kneighbors(features)
    
    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            #if i != j:  # Avoid self-loops
            edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index
    
def build_threshold_knn_graph(features, k=5, threshold_factor=0.5, metric='euclidean'):
    """
    Constructs a k-NN graph with a distance threshold based on min-max distances, storing edge weights.
    
    Parameters:
    - features: Tensor or ndarray of shape (n_samples, d)
    - k: Number of nearest neighbors
    - threshold_factor: Factor to determine threshold as a fraction of (max - min) distance
    - metric: Distance metric used for k-NN
    
    Returns:
    - edge_index: Tensor of shape (2, num_edges) with node connections
    - edge_weights: Tensor of shape (num_edges,) with edge distances
    """
    
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    min_dist, max_dist = distances.min(), distances.max()
    threshold = min_dist + threshold_factor * (max_dist - min_dist)
    
    edge_list = []
    weights = []
    
    for i, (neighbors, dists) in enumerate(zip(indices, distances)):
        for j, dist in zip(neighbors, dists):
            #if i != j and dist <= threshold:  # Avoid self-loops and filter by threshold
            edge_list.append([i, j])
            weights.append(dist)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_weights
    
# Function to create train/val/test masks
def split_data(num_nodes, seed, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."
    
    # Generate random indices for train, val, and test
    indices = np.arange(num_nodes)
    train_idx, temp_idx = train_test_split(indices, test_size=(1 - train_ratio), random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=seed)

    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask
