import unittest
import os
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel
from transformers import AutoModelForCausalLM
import tempfile

from chaiverse.dev.dataset import DatasetLoader, CausalDatasetBuilder
from chaiverse.dev.tokenizer import LlamaTokenizer
from chaiverse.dev.model.lora_model import LoraTrainer


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

            # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
            self.assertTrue(
                os.path.isfile(f"{tmp_dir}/adapter_model.bin"),
                msg=f"{tmp_dir}/adapter_model.bin does not exist",
            )
            self.assertTrue(
                os.path.exists(f"{tmp_dir}/adapter_config.json"),
                msg=f"{tmp_dir}/adapter_config.json does not exist",
            )

    def test_load_pretrained_lora(self):
        r"""
        Check that the model can be saved and loaded properly.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'lora_unittest'
                )
        base_model = tiny_model._load_base_model(load_in_8bit = False)
        tiny_model.instantiate_lora_model(load_in_8bit=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tiny_model.save(path=tmp_dir)
            pretrained_lora_model = PeftModel.from_pretrained(base_model, tmp_dir)

            # check all the weights are the same
            for p1, p2 in zip(tiny_model.model.named_parameters(), pretrained_lora_model.named_parameters()):
                if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                    self.assertTrue(torch.allclose(p1[1], p2[1]), msg=f"{p1[0]} != {p2[0]}")

    def test_continue_training_lora_model(self):
        r"""
        Load peft and checks that it can continue training.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'lora_unittest'
                )
        base_model = tiny_model._load_base_model(load_in_8bit = False)
        tiny_model.instantiate_lora_model(load_in_8bit=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tiny_model.save(path=tmp_dir)
            pretrained_lora_model = PeftModel.from_pretrained(base_model, tmp_dir, is_trainable=True)
            nb_trainable_params = sum(p.numel() for p in pretrained_lora_model.parameters() if p.requires_grad)
            self.assertEqual(nb_trainable_params, 4096)

    def test_merge_model(self):
        r"""
        Check that the model can be merged and saved.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'lora_unittest'
                )
        tiny_model.instantiate_lora_model(load_in_8bit=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tiny_model.save(path=tmp_dir)
            tiny_model.merge(path=tmp_dir)
            tiny_model.model.save_pretrained(tmp_dir)

            self.assertTrue("LlamaForCausalLM" in str(type(tiny_model.model)))

            # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
            self.assertTrue(
                os.path.isfile(f"{tmp_dir}/adapter_model.bin"),
                msg=f"{tmp_dir}/adapter_model.bin does not exist",
            )
            self.assertTrue(
                os.path.exists(f"{tmp_dir}/adapter_config.json"),
                msg=f"{tmp_dir}/adapter_config.json does not exist",
            )
            # check also for `pytorch_model.bin`
            self.assertTrue(
                os.path.exists(f"{tmp_dir}/pytorch_model.bin"),
                msg=f"{tmp_dir}/pytorch_model.bin does not exist",
            )


    def test_load_merge_model(self):
        r"""
        Check that the merged model can be loaded correctly.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'lora_unittest'
                )
        tiny_model.instantiate_lora_model(load_in_8bit=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tiny_model.save(path=tmp_dir)
            tiny_model.merge(path=tmp_dir)
            tiny_model.model.save_pretrained(tmp_dir)

            merged_model = AutoModelForCausalLM.from_pretrained(tmp_dir)

            self.assertTrue("LlamaForCausalLM" in str(type(merged_model)))

            # check all the weights are the same
            for p1, p2 in zip(tiny_model.model.named_parameters(), merged_model.named_parameters()):
                if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                    self.assertTrue(torch.allclose(p1[1], p2[1]), msg=f"{p1[0]} != {p2[0]}")

