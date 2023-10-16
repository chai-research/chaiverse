import unittest
import os

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForCausalLM
import tempfile

from chaiverse.dev.dataset import DatasetLoader, CausalDatasetBuilder
from chaiverse.dev.tokenizer import LlamaTokenizer
from chaiverse.dev.model.lora_model import LoraTrainer

#from .testing_utils import is_peft_available

#@require_peft
class LoraModelTester(unittest.TestCase):
    def setUp(self):
        self.tiny_base_model = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            bias = 'none',
            task_type=TaskType.CAUSAL_LM,
        )

    def test_load_base_model(self):
        r"""
        Simply load a tiny llama base model
        """
        base_model = AutoModelForCausalLM.from_pretrained(
                self.tiny_base_model,
                load_in_8bit=False,
                device_map='auto')

    def test_instantiate_lora_model(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'lora_unittest'
                )
        tiny_model.instantiate_lora_model(load_in_8bit=False)

    def test_check_lora_model_nb_trainable_params(self):
        r"""
        Check that the number of trainable parameters is correct.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'test_peft_model'
                )
        tiny_model.instantiate_lora_model(load_in_8bit=False)
        nb_trainable_params = sum(p.numel() for p in tiny_model.model.parameters() if p.requires_grad)
        self.assertEqual(nb_trainable_params, 4096)

    def test_save_pretrained_lora(self):
        r"""
        Check that the model can be saved and loaded properly.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'lora_unittest'
                )
        tiny_model.instantiate_lora_model(load_in_8bit=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tiny_model.save(path=tmp_dir)

            self.assertTrue(
                os.path.isfile(f"{tmp_dir}/adapter_model.bin"),
                msg=f"{tmp_dir}/adapter_model.bin does not exist",
            )

            self.assertTrue(
                    os.path.exists(f"{tmp_dir}/adapter_config.json"),
                    msg=f"{tmp_dir}/adapter_config.json does not exist",
                    )

    def test_merge_model(self):


