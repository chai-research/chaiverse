from multiprocessing import cpu_count
from functools import partial

import torch
from torch.nn import functional as F

from datasets import concatenate_datasets

from axolotl.utils.distributed import is_main_process

from chaiverse.data_processors.chatml_processor import ChatMLConvoProcessor
from chaiverse.data_processors.utils import get_raw_dataset, tokenize_function


class BaseProcessor:
    def __init__(self, tokenizer, max_length, output_format=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.output_format = output_format
        self.tokenize_func = self._get_tokenize_function()

    def _tokenize(self, dataset, batched=True):
        column_names = dataset.column_names
        tokenized_dataset = dataset.map(
            self.tokenize_func,
            batched=batched,
            num_proc=cpu_count() - 1,
            remove_columns=column_names,
        )
        return tokenized_dataset

    def _get_tokenize_function(self):
        raise NotImplementedError

    def get_tokenized_dataset(self, dataset):
        raise NotImplementedError


class InputOutputProcessor(BaseProcessor):
    def _get_tokenize_function(self):
        func = partial(
            tokenize_function,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return func

    def get_tokenized_dataset(self, dataset):
        return self._tokenize(dataset, batched=True)


class ChatMLProcessor(BaseProcessor):
    def get_tokenized_dataset(self, dataset):
        tokenized_dataset = self._tokenize(dataset, batched=False)
        column_names = tokenized_dataset.column_names
        tokenized_dataset = tokenized_dataset.map(
            self._prepare_tensors,
            remove_columns=column_names,
            num_proc=cpu_count() - 1,
        )
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
        return tokenized_dataset

    def _get_tokenize_function(self):
        processor = ChatMLConvoProcessor(
            self.tokenizer, self.output_format, max_length=self.max_length
        )
        return processor.get_tokenized_convo

    def _padding(self, input_ids, attention_mask, label_mask):
        padding_length = self.max_length - len(input_ids)
        pad_token_id = self.tokenizer.pad_token_id
        if padding_length > 0:
            input_ids = F.pad(input_ids, (0, padding_length), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, padding_length), value=0)
            label_mask = F.pad(label_mask, (0, padding_length), value=0)
        labels = input_ids.clone()
        labels[label_mask == 0] = -100
        return input_ids, attention_mask, labels

    def _prepare_tensors(self, sample):
        input_ids = torch.tensor(sample["input_ids"])
        attention_mask = torch.tensor(sample["attention_mask"])
        label_mask = torch.tensor(sample["label_mask"])
        input_ids, attention_mask, labels = self._padding(
            input_ids, attention_mask, label_mask
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return inputs


class ChatMLDataset:
    def __init__(
        self,
        tokenizer,
        dataset_name,
        dataset_type,
        val_set_size=0.01,
        sequence_len=4096,
    ):
        accepted_formatters = ["chatml", "input_output"]
        assert (
            dataset_type in accepted_formatters
        ), f"Input format must be one of {accepted_formatters}"

        if dataset_type == "chatml":
            self.output_format = "pygmalion"
            self.processor = ChatMLProcessor(
                tokenizer, sequence_len, self.output_format
            )
        else:
            self.output_format = "concatenate"
            self.processor = InputOutputProcessor(
                tokenizer, sequence_len, self.output_format
            )
        self.raw_datasets = get_raw_dataset(
            dataset_name,
            val_set_size,
        )
        self.train_dataset = self.raw_datasets["train"]
        self.eval_dataset = self.raw_datasets["validation"]

    def get_tokenized_dataset(self):
        train_tokenized = self.processor.get_tokenized_dataset(self.train_dataset)
        eval_tokenized = self.processor.get_tokenized_dataset(self.eval_dataset)
        return train_tokenized, eval_tokenized


def prepare_chatml_dataset(cfg, tokenizer):
    lst_train_data = []
    lst_val_data = []
    for data in cfg.datasets:
        dataset = ChatMLDataset(
            tokenizer, data.path, data.type, cfg.val_set_size, cfg.sequence_len
        )
        train_chatml, val_chatml = dataset.get_tokenized_dataset()
        lst_train_data.append(train_chatml)
        lst_val_data.append(val_chatml)
    train_dataset = concatenate_datasets(lst_train_data)
    val_dataset = concatenate_datasets(lst_val_data)
    return train_dataset, val_dataset
