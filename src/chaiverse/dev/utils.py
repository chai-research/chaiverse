from datasets import concatenate_datasets
import numpy as np


def ensure_is_dataset(df):
    if type(df).__name__ == 'DatasetDict':
        df = concatenate_datasets(list(df.values()))
    check_dataset_format(df)
    return df


def check_dataset_format(data):
    assert type(data).__name__ == 'Dataset'


def slice_dataset(df, ixs):
    assert len(ixs) == df.num_rows, 'index has different length with dataset'
    return df.select(np.where(ixs)[0])
