"""Copyright 2024 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import os
from typing import Any, Dict, List

from tabulartextmultimodalfusion.mimic_utils import Discretizer
from tabulartextmultimodalfusion.mimic_utils import get_normalizer
from tabulartextmultimodalfusion.mimic_utils import load_labeled_multimodal_data
from tabulartextmultimodalfusion.mimic_utils import padding_mask
import numpy as np
import omegaconf
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
import transformers

def get_mimic_args():
    """Load MIMIC configuration from mimic_pretrain.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'mimic_pretrain.yaml')
    config = OmegaConf.load(config_path)
    # Override task to finetune since we only support finetune
    config.task = 'finetune'
    return config

class MimicTextTabTimeseries(data.Dataset):
    """Dataset class for MIMIC-IV multimodal data (text, tabular, timeseries only, finetune only)."""
    def __init__(
        self,
        args: omegaconf.DictConfig,
        df: pd.DataFrame,
        tokenizer: transformers.BertTokenizer,
        categorical_var: List[str],
        numerical_var: List[str],
    ):
        if args.task != 'finetune':
            raise ValueError('MimicTextTabTimeseries is only for finetune task.')
        self.args = args
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.categorical_var = categorical_var
        self.numerical_var = numerical_var
        self.labels = df['y_true'].values
        self.discretizer = Discretizer(
            timestep=float(args.timestep),
            store_masks=True,
            impute_strategy=args.impute_strategy,
            start_time=args.start_time,
            config_path=args.discretizer_config_path,
        )
        self.normalizer = get_normalizer(args, self.discretizer)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        item = {}
        # Text
        text = row['text'] if isinstance(row['text'], str) else ''
        encoded = self.tokenizer.encode_plus(
            text[140:],
            max_length=self.args.max_token_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        item['input_ids'] = encoded['input_ids'].squeeze(0)
        item['attention_mask'] = encoded['attention_mask'].squeeze(0)
        # Categorical
        if self.categorical_var:
            categoricals = row[self.categorical_var].values
            categoricals = [str(x) if pd.notnull(x) else '' for x in categoricals]
            try:
                categoricals = [int(x) for x in categoricals]
            except Exception:
                pass
            item['categoricals'] = torch.tensor(categoricals, dtype=torch.long)
        else:
            item['categoricals'] = torch.empty(0, dtype=torch.long)
        # Numerical (static)
        if self.numerical_var:
            numericals = row[self.numerical_var].values.astype(np.float32)
            numericals = torch.tensor(numericals, dtype=torch.float)
        else:
            numericals = torch.empty(0, dtype=torch.float)
        # Timeseries
        ts_filename = row['timeseries']
        if isinstance(ts_filename, str):
            ts_filename = ts_filename.replace('npy', 'csv')
            ts = self._read_timeseries(ts_filename, time_bound=self.args.timeseries_max_seq_len)
            ts_tensor = torch.tensor(ts).float()  # (seq_len, num_features)
            lengths = [ts_tensor.shape[0]]
            pad_mask = padding_mask(
                torch.LongTensor(lengths),
                max_len=self.args.timeseries_max_seq_len,
            ).squeeze(0)  # (seq_len,)
            if pad_mask.sum() > 0:
                ts_pooled = ts_tensor[pad_mask].mean(dim=0)
            else:
                ts_pooled = torch.zeros(ts_tensor.shape[1], dtype=torch.float)
        else:
            ts_pooled = torch.zeros(self.args.timeseries_input_dim, dtype=torch.float)
        if numericals.numel() > 0:
            numerical = torch.cat([numericals, ts_pooled], dim=0)
        else:
            numerical = ts_pooled
        item['numerical'] = numerical
        item['labels'] = torch.tensor(row['y_true'], dtype=torch.long)
        return item

    def _read_timeseries(self, ts_filename, time_bound=None):
        ret = []
        with open(ts_filename, 'r') as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == 'Hours'
            for line in tsfile:
                mas = line.strip().split(',')
                if time_bound is not None:
                    t = float(mas[0])
                    if t > time_bound + 1e-6:
                        break
                ret.append(np.array(mas))
        ret = np.stack(ret)
        data_discretized, _ = self.discretizer.transform(
            ret, end=self.args.timeseries_max_seq_len
        )
        data_discretized = self.normalizer.transform(data_discretized)
        return data_discretized

    def __len__(self) -> int:
        return len(self.df)

def mimic_to_tensor_dataset(mimic_dataset):
    input_ids = []
    categoricals = []
    numericals = []
    labels = []
    attention_masks = []
    for i in range(len(mimic_dataset)):
        item = mimic_dataset[i]
        input_ids.append(item['input_ids'].unsqueeze(0))
        categoricals.append(item['categoricals'].unsqueeze(0))
        numericals.append(item['numerical'].unsqueeze(0))
        labels.append(item['labels'].unsqueeze(0))
        attention_masks.append(item['attention_mask'].unsqueeze(0))
    input_ids = torch.cat(input_ids, dim=0)
    categoricals = torch.cat(categoricals, dim=0)
    numericals = torch.cat(numericals, dim=0)
    labels = torch.cat(labels, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return TensorDataset(input_ids, categoricals, numericals, labels, attention_masks)

def load_mimic(
    args: omegaconf.DictConfig, tokenizer: transformers.AutoTokenizer
) -> Dict[str, Any]:
    """Load MIMIC dataset (text, tabular, timeseries only, finetune only)."""
    if args.task != 'finetune':
        raise ValueError('Only finetune task is supported for MIMIC.')
    dataframes = load_labeled_multimodal_data(args)
    df_train = dataframes['train']
    y_col = 'y_true'
    exclude_cols = set(['text', 'timeseries', y_col, 'split', 'stay_id', 'subject_id', 'dicom_id', 'study_id', 'stay'])
    categorical_var = [col for col in df_train.columns if (df_train[col].dtype == 'object' or df_train[col].dtype.name == 'category') and col not in exclude_cols]
    numerical_var = [col for col in df_train.columns if (np.issubdtype(df_train[col].dtype, np.number) and col not in exclude_cols)]
    dataset = create_multimodal_dataset_from_dataframes(
        args, dataframes, tokenizer, categorical_var, numerical_var
    )
    # Convert to TensorDataset for compatibility
    dataset['train'] = mimic_to_tensor_dataset(dataset['train'])
    dataset['valid'] = mimic_to_tensor_dataset(dataset['valid'])
    dataset['test'] = mimic_to_tensor_dataset(dataset['test'])
    dataset['categorical_var'] = categorical_var
    dataset['numerical_var'] = numerical_var
    dataset['num_cat_var'] = len(categorical_var)
    dataset['cat_vocab_sizes'] = [df_train[col].nunique() for col in categorical_var]
    dataset['num_numerical_var'] = len(numerical_var) + args.timeseries_input_dim
    dataset['y_col'] = y_col
    return dataset

def create_multimodal_dataset_from_dataframes(
    args: omegaconf.DictConfig,
    dataframes: Dict[str, pd.DataFrame],
    tokenizer: Any,
    categorical_var: List[str],
    numerical_var: List[str],
) -> Dict[str, Any]:
    """Creates datasets for training, validation, and testing from DataFrames (text, tabular, timeseries only)."""
    mm_train = MimicTextTabTimeseries(
        args=args,
        df=dataframes['train'],
        tokenizer=tokenizer,
        categorical_var=categorical_var,
        numerical_var=numerical_var,
    )
    mm_test = MimicTextTabTimeseries(
        args=args,
        df=dataframes['test'],
        tokenizer=tokenizer,
        categorical_var=categorical_var,
        numerical_var=numerical_var,
    )
    mm_valid = MimicTextTabTimeseries(
        args=args,
        df=dataframes['valid'],
        tokenizer=tokenizer,
        categorical_var=categorical_var,
        numerical_var=numerical_var,
    )
    return {
        'train': mm_train,
        'valid': mm_valid,
        'test': mm_test,
        'tabular_data_information': dataframes['tabular_data_information'],
    }