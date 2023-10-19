import copy

import datasets
from datasets import Value
import numpy as np


def load_dataset(path, **kw):
    df = datasets.load_dataset(path, **kw)
    df = ensure_is_dataset(df)
    return df


def ensure_is_dataset(df):
    if type(df).__name__ == 'DatasetDict':
        df = datasets.concatenate_datasets(list(df.values()))
    check_dataset_format(df)
    return df


def check_dataset_format(data):
    assert type(data).__name__ == 'Dataset'


def slice_dataset(df, ixs):
    assert len(ixs) == df.num_rows, 'index has different length with dataset'
    return df.select(np.where(ixs)[0])


def ensure_is_list(obj):
    return obj if isinstance(obj, list) else [obj]


def format_dataset_dtype(data, column, dtype):
    data = copy.copy(data)
    if type(data).__name__ == 'DatasetDict':
        for fold, df in data.items():
            data[fold] = _format_dataset_dtype(df, column, dtype)
    else:
        data = _format_dataset_dtype(data, column, dtype)
    return data


def _format_dataset_dtype(df, column, dtype):
    features = df.features.copy()
    if features[column].dtype != dtype:
        features[column] = Value(dtype)
        df = df.cast(features)
    return df
