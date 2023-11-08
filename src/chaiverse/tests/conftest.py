import datasets
import pytest


@pytest.fixture()
def hf_dataset():
    train_txt = {
            'input_text': ['a', 'b', 'c', 'd', 'e', 'f'],
            'output_text': ['f', 'g', 'h', 'i', 'j', 'k'],
            'labels': [1, 0, 1, 0, 1, 0],
            }
    val_txt = {
            'input_text': ['o', 'p', 'q', 'r'],
            'output_text': ['w', 'x', 'y', 'z'],
            'labels': [1, 1, 0, 0],
            }

    return datasets.DatasetDict({
        'train': datasets.Dataset.from_dict(train_txt),
        'validation': datasets.Dataset.from_dict(val_txt),
        })
