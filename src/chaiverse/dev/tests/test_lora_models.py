import unittest

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForCausalLM

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

    def test_instantiate_lora_model(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        tiny_model = LoraTrainer(
                model_name = self.tiny_base_model,
                output_dir = 'test_peft_model'
                )
        tiny_model.instantiate_lora_model(load_in_8bit=False)
        #pretrained_model = tiny_model._load_base_model(load_in_8bit=False)
        #pretrained_model = prepare_model_for_int8_training(pretrained_model)
        #tiny_model.model = tiny_model._load_lora_model(pretrained_model)
        #tiny_model.model.print_trainable_parameters()
