"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import importlib
import logging
import os
import random
import signal
import sys
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fire
import torch
import yaml

# add src to the pythonpath so we don't need to pip install this
from optimum.bettertransformer import BetterTransformer
from transformers import GenerationConfig, TextStreamer

from axolotl.logging_config import configure_logging
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.data import prepare_dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.tokenization import check_dataset_labels
from axolotl.utils.trainer import setup_trainer, calculate_total_num_steps
from axolotl.utils.wandb import setup_wandb_env_vars

from chaiverse.data_processors.dialogue_dataset import prepare_chatml_dataset
from datasets import concatenate_datasets

configure_logging()
LOG = logging.getLogger("axolotl.scripts")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def get_multi_line_input() -> Optional[str]:
    print("Give me an instruction (Ctrl + D to finish): ")
    instruction = ""
    for line in sys.stdin:
        instruction += line  # pylint: disable=consider-using-join
    return instruction


def choose_config(path: Path):
    yaml_files = list(path.glob("*.yml"))

    if not yaml_files:
        raise ValueError(
            "No YAML config files found in the specified directory. Are you using a .yml extension?"
        )

    print("Choose a YAML file:")
    for idx, file in enumerate(yaml_files):
        print(f"{idx + 1}. {file}")

    chosen_file = None
    while chosen_file is None:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(yaml_files):
                chosen_file = yaml_files[choice - 1]
            else:
                print("Invalid choice. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return chosen_file


def check_not_in(list1: List[str], list2: Union[Dict[str, Any], List[str]]) -> bool:
    return not any(el in list2 for el in list1)


def train(
    config: Path = Path("configs/"),
    prepare_ds_only: bool = False,
    **kwargs,
):
    if Path(config).is_dir():
        config = choose_config(config)

    # load the config from the yaml file
    with open(config, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    validate_config(cfg)

    normalize_config(cfg)

    setup_wandb_env_vars(cfg)

    # load the tokenizer first
    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)
    if (
        check_not_in(["shard", "merge_lora"], kwargs) and not cfg.inference
    ):  # don't need to load dataset for these
        load_by_chaiverse = []
        load_by_axo = []

        new_config = copy.deepcopy(cfg)
        for item in cfg["datasets"]:
            if item["type"] == "chatml" or item["type"] == "input_output":
                load_by_chaiverse.append(item)
            else:
                load_by_axo.append(item)
        new_config["datasets"] = load_by_chaiverse
        cfg["datasets"] = load_by_axo
        if load_by_chaiverse:
            train_chaiverse, eval_chaiverse = prepare_chatml_dataset(new_config, tokenizer)
            total_num_steps_chaiverse = calculate_total_num_steps(
                new_config, train_chaiverse, tokenizer
            )
        if load_by_axo:
            train_dataset, eval_dataset, total_num_steps = prepare_dataset(
                cfg, tokenizer
            )
        if load_by_chaiverse and load_by_axo:
            train_dataset = concatenate_datasets([train_dataset, train_chaiverse])
            eval_dataset = concatenate_datasets([eval_dataset, eval_chaiverse])
            total_num_steps = total_num_steps + total_num_steps_chaiverse
        if not load_by_axo:
            train_dataset = train_chaiverse
            eval_dataset = eval_chaiverse
            total_num_steps = total_num_steps_chaiverse

    if cfg.debug or "debug" in kwargs:
        LOG.info("check_dataset_labels...")
        check_dataset_labels(
            train_dataset.select(
                [random.randrange(0, len(train_dataset) - 1) for _ in range(5)]  # nosec
            ),
            tokenizer,
        )

    if prepare_ds_only:
        LOG.info("Finished preparing dataset. Exiting...")
        return

    # Load the model and tokenizer
    LOG.info("loading model and (optionally) peft_config...")
    model, peft_config = load_model(cfg, tokenizer)

    safe_serialization = cfg.save_safetensors is True

    if "merge_lora" in kwargs and cfg.adapter is not None:
        LOG.info("running merge of LoRA with base model")
        model = model.merge_and_unload()
        model.to(dtype=torch.float16)

        if cfg.local_rank == 0:
            LOG.info("saving merged model")
            model.save_pretrained(
                str(Path(cfg.output_dir) / "merged"),
                safe_serialization=safe_serialization,
            )
            tokenizer.save_pretrained(str(Path(cfg.output_dir) / "merged"))
        return

    if "shard" in kwargs:
        model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)
        return

    trainer = setup_trainer(
        cfg, train_dataset, eval_dataset, model, tokenizer, total_num_steps
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        LOG.info("Compiling torch model")
        model = torch.compile(model)

    # go ahead and presave, so we have the adapter config available to inspect
    if peft_config:
        LOG.info(f"Pre-saving adapter config to {cfg.output_dir}")
        peft_config.save_pretrained(cfg.output_dir)

    # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
    if cfg.local_rank == 0:

        def terminate_handler(_, __, model):
            if cfg.flash_optimum:
                model = BetterTransformer.reverse(model)
            model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)
            sys.exit(0)

        signal.signal(
            signal.SIGINT, lambda signum, frame: terminate_handler(signum, frame, model)
        )

    LOG.info("Starting trainer...")
    if cfg.group_by_length:
        LOG.info("hang tight... sorting dataset for group_by_length")
    resume_from_checkpoint = cfg.resume_from_checkpoint
    if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints:
        possible_checkpoints = [
            str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")
        ]
        if len(possible_checkpoints) > 0:
            sorted_paths = sorted(
                possible_checkpoints,
                key=lambda path: int(path.split("-")[-1]),
            )
            resume_from_checkpoint = sorted_paths[-1]
            LOG.info(
                f"Using Auto-resume functionality to start with checkpoint at {resume_from_checkpoint}"
            )

    if not Path(cfg.output_dir).is_dir():
        os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer.save_pretrained(cfg.output_dir)
    if cfg.flash_optimum:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    LOG.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")

    # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
    # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
    if cfg.fsdp:
        trainer.save_model(cfg.output_dir)
    elif cfg.local_rank == 0:
        if cfg.flash_optimum:
            model = BetterTransformer.reverse(model)
        model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)
    return model, tokenizer


if __name__ == "__main__":
    fire.Fire(train)
