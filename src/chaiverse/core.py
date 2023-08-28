import yaml
import os
from pathlib import Path

import chaiverse as cv
from axolotl.utils.models import load_model, load_tokenizer


class ChaiLLM():
    def __init__(self):
        self._output_dir = None
        self._config_file_path = None

    def fit(
            self,
            dataset,
            output_dir,
            num_epochs=1,
            eval_steps=200,
            learning_rate=2e-5,
            wandb_project=None,
            wandb_entity=None,
            sequence_len=1024,
            logging_steps=1,
            val_set_size=0.01,
            gradient_accumulation_steps=2,
            micro_batch_size=4,
            bf16=True,
            gradient_checkpointing=True,
            ):
        self._set_output_dir(output_dir)
        self._save_yaml_file(
            base_model=self.model_url,
            base_model_config=self.tokenizer_url,
            datasets=[{'path': dataset.repo_url, 'type': dataset.data_type}],
            dataset_prepared_path='last_run_prepared',
            val_set_size=val_set_size,
            output_dir=self.output_dir,
            sequence_len=sequence_len,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            gradient_accumulation_steps=gradient_accumulation_steps,
            micro_batch_size=micro_batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=logging_steps,
            flash_attention=self.use_flash_attention,
            eval_steps=eval_steps
        )
        model, tokenizer = cv.trainer.train(self.config_file_path)
        self.model = model
        self.tokenizer = tokenizer

    def push_to_hub(self, model_url, private):
        self.model.push_to_hub(model_url, private)
        self.tokenizer.push_to_hub(model_url, private)

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def config_file_path(self):
        return self._config_file_path

    def _set_output_dir(self, output_dir):
        cv.utils.ensure_dir_exists(output_dir)
        self._output_dir = os.path.abspath(output_dir)

    def _set_config_file_path(self, path):
        self._config_file_path = path
    
    def _save_yaml_file(self, **configs):
        path = os.path.join(self.output_dir, 'trainer_config.yaml')
        with open(path, 'w') as f:
            yaml.dump(configs, f)
        self._set_config_file_path(path)


class LLaMA7b(ChaiLLM):
    @property
    def model_url(self):
        return 'NousResearch/Llama-2-7b-hf'

    @property
    def tokenizer_url(self):
        return 'NousResearch/Llama-2-7b-hf'

    @property
    def use_flash_attention(sefl):
        return True


# NO IDEA why this throws error, bypassing accelerate for now as its command line anyway...
def launch_training(config_path):
    import subprocess
    root_dir = Path(os.path.dirname(cv.__file__))
    execution_path = root_dir / "trainer.py"
    command = f"accelerate launch {execution_path} {config_path}"
    subprocess.check_output(command, shell=True)
