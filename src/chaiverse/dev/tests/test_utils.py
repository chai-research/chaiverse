from unittest import mock
import datasets
import pytest
import numpy as np

from chaiverse.dev import utils


@pytest.fixture()
def hf_dataset():
    train_txt = {
            'input_text': ['a', 'b', 'c', 'd', 'e', 'f'],
            'output_text': ['f', 'g', 'h', 'i', 'j', 'k']}
    val_txt = {
            'input_text': ['o', 'p', 'q', 'r'],
            'output_text': ['w', 'x', 'y', 'z']}

    return datasets.DatasetDict({
        'train': datasets.Dataset.from_dict(train_txt),
        'validation': datasets.Dataset.from_dict(val_txt),
        })


def test_ensure_is_dataset(hf_dataset):
    data = utils.ensure_is_dataset(hf_dataset)
    assert data['input_text'] == ['a', 'b', 'c', 'd', 'e', 'f', 'o', 'p', 'q', 'r']
    assert data['output_text'] == ['f', 'g', 'h', 'i', 'j', 'k', 'w', 'x', 'y', 'z']


def test_slice_dataset(hf_dataset):
    data = hf_dataset['train']
    ixs = np.array([1, 0]).astype(bool)
    with pytest.raises(AssertionError) as e:
        utils.slice_dataset(data, ixs)
    assert 'different length' in str(e)

    ixs = np.full(len(data), False)
    ixs[-1] = True
    sliced_data = utils.slice_dataset(data, ixs)
    assert sliced_data.to_dict() == {'input_text': ['f'], 'output_text': ['k']}
