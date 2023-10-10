from typing import Union

from transformers import AutoTokenizer
import transformers


LLAMA_DEFAULT_EOS_TOKEN = "</s>"
LLAMA_DEFAULT_TOKENIZE_MODEL = 'NousResearch/Llama-2-7b-hf'
LLAMA_DEFAULT_TOKENIZE_TYPE = 'LlamaTokenizer'


class Tokenizer:
    _tokenizer_cls = AutoTokenizer

    def __init__(
            self,
            base_model: str,
            tokenizer_type: Union[str, None] = None,
            tokenizer_user_fast: bool = True,
            tokenizer_special_tokens: Union[dict, None] = None,
            padding_side: str = 'left',
            truncation_side: str = 'left',
            ):
        self.base_model = base_model
        self.tokenizer_type = tokenizer_type
        self.tokenizer_user_fast = tokenizer_user_fast
        self.tokenizer_special_tokens = tokenizer_special_tokens
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        if tokenizer_type:
            self._tokenizer_cls = getattr(transformers, tokenizer_type)

    def load(self, **kwargs):
        tokenizer = self._tokenizer_cls.from_pretrained(
                self.base_model,
                use_fast=self.tokenizer_user_fast,
                padding_side=self.padding_side,
                truncation_side=self.truncation_side,
                **kwargs)
        tokenizer = self._add_special_tokens(tokenizer)
        return tokenizer

    def _add_special_tokens(self, tokenizer):
        if self.tokenizer_special_tokens:
            for key, token in self.tokenizer_special_tokens.items():
                tokenizer.add_special_tokens({key: token})
        return tokenizer


class LlamaTokenizer(Tokenizer):

    def __init__(
            self,
            base_model: str = LLAMA_DEFAULT_TOKENIZE_MODEL,
            tokenizer_type: str = LLAMA_DEFAULT_TOKENIZE_TYPE,
            tokenizer_user_fast: bool = True,
            tokenizer_special_tokens: Union[dict, None] = None,
            padding_side: str = 'left',
            truncation_side: str = 'left',
            ):
        super(LlamaTokenizer, self).__init__(
                base_model,
                tokenizer_type,
                tokenizer_user_fast,
                tokenizer_special_tokens,
                padding_side,
                truncation_side,
                )

    def _format_llama_pad_token(self, tokenizer):
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN
        return tokenizer

    def _add_special_tokens(self, tokenizer):
        tokenizer = self._format_llama_pad_token(tokenizer)
        if self.tokenizer_special_tokens:
            for key, token in self.tokenizer_special_tokens.items():
                tokenizer.add_special_tokens({key: token})
        return tokenizer
