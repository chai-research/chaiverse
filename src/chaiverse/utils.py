from functools import wraps
import copy

import datasets
from datasets import Value
import numpy as np


def datasetdict_inplace_wrapper(func):
    @wraps(func)
    def wrapped_func(df, *args, **kwargs):
        df = copy.copy(df)
        if type(df).__name__ == 'DatasetDict':
            for fold, data in df.items():
                df[fold] = func(df[fold], *args, **kwargs)
        else:
            df = func(df, *args, **kwargs)
        return df
    return wrapped_func


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


@datasetdict_inplace_wrapper
def format_dataset_dtype(df, column, dtype):
    features = df.features.copy()
    if features[column].dtype != dtype:
        features[column] = Value(dtype)
        df = df.cast(features)
    return df
