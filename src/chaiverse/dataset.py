from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Union

import datasets
from chaiverse import utils
from chaiverse.logging_utils import logging_manager


class DatasetLoader:

    @logging_manager('dataset_jobs')
    def __init__(
            self,
            hf_path,
            data_samples: Union[int, None] = None,
            validation_split_size: Union[int, float] = 0,
            shuffle: bool = False,
            seed: int = 1):
        self.hf_path = hf_path
        self.data_samples = data_samples
        self.validation_split_size = validation_split_size
        self.shuffle = shuffle
        self.seed = seed

    def load(self):
        df = self._load_df()
        if self.shuffle:
            df = self.shuffle_df(df)
        if self.data_samples:
            df = self.sample_df(df)
        if self.validation_split_size != 0:
            df = self.validation_split(df)
        return df

    def shuffle_df(self, df):
        df = df.shuffle(seed=self.seed)
        return df

    def sample_df(self, df):
        for fold in df:
            sub_df = df[fold]
            n_samples = min(len(sub_df), self.data_samples)
            df[fold] = sub_df.select(range(n_samples))
        return df

    def validation_split(self, df):
        df = utils.ensure_is_dataset(df)
        df = df.train_test_split(
                test_size=self.validation_split_size,
                shuffle=False)
        df['validation'] = df.pop('test')
        return df

    def _load_df(self):
        df = datasets.load_dataset(self.hf_path)
        return df


class BaseDatasetBuilder(metaclass=ABCMeta):

    def __init__(
            self,
            tokenizer_loader,
            block_size=2048,
            input_column='input_text',
            output_column='output_text',
            ):
        self.block_size = block_size
        self.tokenizer_loader = tokenizer_loader
        self.input_column = input_column
        self.output_column = output_column

    def generate(self, df, n_jobs=1):
        self.tokenizer = self.tokenizer_loader.load()
        self._check_model_max_length()
        data = self.apply(df, n_jobs=n_jobs)
        return data

    def apply(self, df, n_jobs=1):
        tokenized_data = df.map(
                self._process_data,
                batched=True,
                num_proc=n_jobs)
        return tokenized_data

    @abstractmethod
    def _process_data(self, data):
        raise NotImplementedError

    def _get_joint_input_output_texts(self, data):
        input_texts, output_texts = data[self.input_column], data[self.output_column]
        joint_data = [input_ + output_ for input_, output_ in zip(input_texts, output_texts)]
        return joint_data

    def _padding_tokenize(self, data):
        tokenized_data = self.tokenizer(
            data,
            padding="max_length",
            max_length=self.block_size,
            return_token_type_ids=False,
            truncation=True)
        return tokenized_data

    def _check_model_max_length(self):
        max_length = self.tokenizer.model_max_length
        assert self.block_size <= max_length, f'Block size exceeds model_max_length: {max_length}'


class CausalDatasetBuilder(BaseDatasetBuilder):

    def __init__(
            self,
            tokenizer_loader,
            block_size=2048,
            input_column='input_text',
            output_column='output_text',
            input_mask_label=-100,
            ):
        super().__init__(tokenizer_loader, block_size, input_column, output_column)
        self.input_mask_label = input_mask_label

    def _process_data(self, data):
        joint_data = self._get_joint_input_output_texts(data)
        tokenized_data = self._padding_tokenize(joint_data)
        output_lengths = self._get_output_token_lengths(data)
        tokenized_data = self._add_input_masked_labels(tokenized_data, output_lengths)
        return tokenized_data

    def _get_output_token_lengths(self, data):
        token_lengths = [len(self.tokenizer(txt).input_ids) for txt in data[self.output_column]]
        return token_lengths

    def _add_input_masked_labels(self, data, output_lengths):
        data['labels'] = deepcopy(data.input_ids)
        num_rows = len(data['labels'])
        for row in range(num_rows):
            joint_length = len(data['labels'][row])
            output_length = output_lengths[row]
            input_length = joint_length - output_length
            for idx in range(0, input_length):
                data['labels'][row][idx] = self.input_mask_label
        return data


class RewardDatasetBuilder(BaseDatasetBuilder):

    def __init__(
            self,
            tokenizer_loader,
            block_size=1024,
            input_column='input_text',
            output_column=None,
            label_column=None,
            ):
        super().__init__(tokenizer_loader, block_size, input_column, output_column)
        self.label_column = label_column

    def _process_data(self, data):
        input_data = self._format_input_data(data)
        tokenized_data = self._padding_tokenize(input_data)
        if self.label_column:
            tokenized_data = self._add_label_column(data, tokenized_data)
        return tokenized_data

    def _format_input_data(self, data):
        if self.output_column:
            input_data = self._get_joint_input_output_texts(data)
        else:
            input_data = data[self.input_column]
        return input_data

    def _add_label_column(self, data, tokenized_data):
        tokenized_data['labels'] = data[self.label_column]
        return tokenized_data
