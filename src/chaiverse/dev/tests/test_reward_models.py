import os
import torch
import pytest

import tempfile
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from torch import nn
from chaiverse.dev.dataset import DatasetLoader, RewardDatasetBuilder
from chaiverse.dev.tokenizer import GPT2Tokenizer
from chaiverse.dev.model.reward_model import RewardClassificationTrainer

@pytest.fixture
def tiny_base_model_id():
    return "hf-internal-testing/tiny-random-gpt2"

@pytest.fixture
def tiny_base_model(tiny_base_model_id):
    return AutoModelForSequenceClassification.from_pretrained(
                tiny_base_model_id,
                num_labels=1,
                device_map='cpu',
                )

@pytest.fixture
def tokenize_loader():
    return GPT2Tokenizer(
            padding_side='right',
            truncation_side='left',
            )

@pytest.fixture
def tiny_model(tiny_base_model_id,tokenize_loader):
    tiny_model = RewardClassificationTrainer(
            model_name=tiny_base_model_id,
            tokenize_loader=tokenize_loader,
            device_map = "cpu",
            output_dir='test_reward_model',
            learning_rate=1e-5,
            num_train_epochs=1,
            bf16=False,
            eval_strategy='steps',
            eval_steps=50,
            )
    tiny_model.instantiate_reward_model()
    return tiny_model

def test_load_base_model(tiny_base_model):
    assert tiny_base_model is not None

def test_instantiate_reward_model(tiny_model):
    assert tiny_model.model is not None
    assert "GPT2ForSequenceClassification" in str(type(tiny_model.model))

def test_check_reward_model_nb_trainable_params(tiny_model):
    r"""
    Check that the number of trainable parameters is correct.
    """
    nb_trainable_params = sum(p.numel() for p in tiny_model.model.parameters() if p.requires_grad)
    assert nb_trainable_params == 112032
