import pytest
import os
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel
from transformers import AutoModelForCausalLM
import tempfile

from chaiverse.dataset import DatasetLoader, CausalDatasetBuilder
from chaiverse.tokenizer import LlamaTokenizer
from chaiverse.model.lora_model import LoraTrainer

@pytest.fixture
def tiny_base_model_id():
    return "HuggingFaceH4/tiny-random-LlamaForCausalLM"

@pytest.fixture
def lora_config():
    return LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                bias = 'none',
                task_type=TaskType.CAUSAL_LM,
            )

@pytest.fixture
def tiny_base_model(tiny_base_model_id):
    r"""
    Simply load a tiny llama base model
    """
    return AutoModelForCausalLM.from_pretrained(
            tiny_base_model_id,
            load_in_8bit=False,
            device_map='cpu')

@pytest.fixture
def tiny_model(tiny_base_model_id):
    r"""
    Simply creates a peft model and checks that it can be loaded.
    """
    tiny_model = LoraTrainer(
            model_name = tiny_base_model_id,
            output_dir = 'lora_unittest',
            device_map = 'cpu'
            )
    tiny_model.instantiate_lora_model(load_in_8bit=False)
    return tiny_model

@pytest.fixture
def test_load_base_model(tiny_base_model):
    assert tiny_base_model is not None

def test_instantiate_lora_model(tiny_model):
    assert tiny_model.model is not None

def test_check_lora_model_nb_trainable_params(tiny_model):
    r"""
    Check that the number of trainable parameters is correct.
    """
    nb_trainable_params = sum(p.numel() for p in tiny_model.model.parameters() if p.requires_grad)
    assert nb_trainable_params == 4096

def test_save_pretrained_lora(tiny_model):
    r"""
    Check that the model can be saved and loaded properly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_model.save(path=tmp_dir)

        # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
        assert os.path.isfile(f"{tmp_dir}/adapter_model.bin"), f"{tmp_dir}/adapter_model.bin does not exist"
        assert os.path.exists(f"{tmp_dir}/adapter_config.json"), f"{tmp_dir}/adapter_config.json does not exist"

def test_load_pretrained_lora(tiny_model, tiny_base_model):
    r"""
    Check that the model can be saved and loaded properly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_model.save(path=tmp_dir)
        pretrained_lora_model = PeftModel.from_pretrained(tiny_base_model, tmp_dir)

        # check all the weights are the same
        for p1, p2 in zip(tiny_model.model.named_parameters(), pretrained_lora_model.named_parameters()):
            if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                assert torch.allclose(p1[1], p2[1]), f"{p1[0]} != {p2[0]}"

def test_continue_training_lora_model(tiny_model, tiny_base_model):
    r"""
    Load peft and checks that it can continue training.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_model.save(path=tmp_dir)
        pretrained_lora_model = PeftModel.from_pretrained(tiny_base_model, tmp_dir, is_trainable=True)
        nb_trainable_params = sum(p.numel() for p in pretrained_lora_model.parameters() if p.requires_grad)
        assert nb_trainable_params == 4096

def test_merge_model(tiny_model):
    r"""
    Check that the model can be merged and saved.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_model.save(path=tmp_dir)
        tiny_model.merge(path=tmp_dir)
        tiny_model.model.save_pretrained(tmp_dir+'/merged')

        assert "LlamaForCausalLM" in str(type(tiny_model.model))

        # check that the files `generation_config.json`, `adapter_config.json`,
        # 'pytorch_model.bin' are in the directory
        assert os.path.isfile(f"{tmp_dir}/merged/generation_config.json"), f"{tmp_dir}/merged/generation_config does not exist"
        assert os.path.exists(f"{tmp_dir}/merged/config.json"), f"{tmp_dir}/merged/config.json does not exist"
        assert os.path.exists(f"{tmp_dir}/merged/pytorch_model.bin"), f"{tmp_dir}/merged/pytorch_model.bin does not exist"

def test_load_merge_model(tiny_model):
    r"""
    Check that the merged model can be loaded correctly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_model.save(path=tmp_dir)
        tiny_model.merge(path=tmp_dir)
        tiny_model.model.save_pretrained(tmp_dir+'/merged')

        merged_model = AutoModelForCausalLM.from_pretrained(tmp_dir+'/merged')

        assert "LlamaForCausalLM" in str(type(merged_model))

        # check all the weights are the same
        for p1, p2 in zip(tiny_model.model.named_parameters(), merged_model.named_parameters()):
            if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                assert torch.allclose(p1[1], p2[1]), f"{p1[0]} != {p2[0]}"

