import os
import yaml

from datasets import load_dataset as load_hf_dataset
from axolotl.utils.dict import DictDefault


def ensure_dir_exists(path):
    if '.' in os.path.basename(path):
        directory = os.path.dirname(path)
    else:
        directory = path
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_dataset(repo_url, data_type):
    assert data_type in {'chatml', 'input_output'}, 'Unsupported dataset format'
    dataset = load_hf_dataset(repo_url)
    dataset.repo_url = repo_url
    dataset.data_type = data_type
    return dataset


def load_config(config_file_path):
    with open(config_file_path, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))
    return cfg
