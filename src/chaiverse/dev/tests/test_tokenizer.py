import pytest

from chaiverse.dev.tokenizer import Tokenizer, LlamaTokenizer


def test_dafult_tokenizer():
    tk = Tokenizer(base_model='gpt2')
    assert tk._tokenizer_cls.__name__ == 'AutoTokenizer'
    tokenizer = tk.load()
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == 'left'


def test_tokenizer_with_special_tokens():
    tk = Tokenizer(
            base_model='gpt2',
            tokenizer_special_tokens={'bos_token': 'test1', 'eos_token': 'test2'})
    tokenizer = tk.load()
    assert tokenizer.bos_token == 'test1'
    assert tokenizer.eos_token == 'test2'


def test_llama_tokenizer_default_pad_token():
    tk = LlamaTokenizer()
    tokenizer = tk.load()
    assert tokenizer.pad_token == '</s>'
