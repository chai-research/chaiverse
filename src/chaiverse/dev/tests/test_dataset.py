from mock import Mock, patch
from unittest import mock
import datasets
import pytest

from chaiverse.dev.dataset import DatasetLoader


@pytest.fixture(autouse='session')
def mock_request():
    with patch('chaiverse.dev.logging_utils.requests.post', Mock()) as request:
        yield request


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


@pytest.fixture()
def mock_load_dataset(hf_dataset):
    data = mock.Mock(return_value=hf_dataset)
    with mock.patch('datasets.load_dataset', data):
        yield data


def test_datasetloader_load_default_df(mock_load_dataset):
    data_loader = DatasetLoader(hf_path='')
    df = data_loader.load()
    assert df['train']['input_text'] == ['a', 'b', 'c', 'd', 'e', 'f']
    assert df['train']['output_text'] == ['f', 'g', 'h', 'i', 'j', 'k']
    assert df['validation']['input_text'] == ['o', 'p', 'q', 'r']
    assert df['validation']['output_text'] == ['w', 'x', 'y', 'z']
    assert df.shape == {'train': (6, 2), 'validation': (4, 2)}


def test_datasetloader_shuffle_df(mock_load_dataset):
    data_loader = DatasetLoader(hf_path='', shuffle=True, seed=2)
    df = data_loader.load()
    assert df['train']['input_text'] != ['a', 'b', 'c', 'd', 'e', 'f']
    assert df['validation']['input_text'] != ['f', 'g', 'h', 'i', 'j', 'k']

    data_loader = DatasetLoader(hf_path='', shuffle=True, seed=2)
    df1 = data_loader.load()
    assert df['train'].to_dict() == df1['train'].to_dict()

    data_loader = DatasetLoader(hf_path='', shuffle=True, seed=3)
    df2 = data_loader.load()
    assert df['train'].to_dict() != df2['train'].to_dict()


def test_datasetloader_sample_df(mock_load_dataset):
    data_loader = DatasetLoader(
            hf_path='',
            data_samples=2,
            )
    df = data_loader.load()
    assert df.shape == {'train': (2, 2), 'validation': (2, 2)}
    assert df['train']['input_text'] == ['a', 'b']
    assert df['validation']['input_text'] == ['o', 'p']


def test_datasetloader_sample_df_with_large_n_samples(mock_load_dataset):
    data_loader = DatasetLoader(
            hf_path='',
            data_samples=50,
            )
    df = data_loader.load()
    assert df.shape == {'train': (6, 2), 'validation': (4, 2)}


def test_datasetloader_validation_split(mock_load_dataset):
    data_loader = DatasetLoader(
            hf_path='',
            validation_split_size=0.1,
            )
    df = data_loader.load()
    assert df.shape == {'train': (9, 2), 'validation': (1, 2)}
    assert df['validation']['input_text'] == ['r']
    assert df['validation']['output_text'] == ['z']
