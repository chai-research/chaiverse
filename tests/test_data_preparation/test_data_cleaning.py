import pytest

import datasets
from chaiverse.data_preparation import data_cleaning
from chaiverse import utils


def test_slice_columns(hf_dataset):
    cols = ['input_text', 'output_text']
    res = data_cleaning.slice_columns(hf_dataset, cols)
    assert res.column_names == {'train': cols, 'validation': cols}

    hf_dataset = utils.ensure_is_dataset(hf_dataset)
    res = data_cleaning.slice_columns(hf_dataset, cols)
    assert res.column_names == cols


def test_slice_columns_raise_with_wrong_column_names(hf_dataset):
    with pytest.raises(AssertionError) as e:
        data_cleaning.slice_columns(hf_dataset, 'bad_name')
        assert 'is not subset of data columns' in str(e)


def test_slice_columns_with_string_name(hf_dataset):
    res = data_cleaning.slice_columns(hf_dataset, 'labels')
    assert res.column_names == {'train': ['labels'], 'validation': ['labels']}


def test_filter_by_ending_with_bot_response():
    txt = {
            'input_text': [
                '123\nuser:',
                '234\nbot:',
                '345\nuser:']}

    df = datasets.Dataset.from_dict(txt)
    df = data_cleaning.filter_by_ending_with_bot_response(df, col='input_text', user_label='user:')
    assert len(df) == 1
    assert df['input_text'] == ['234\nbot:']


def test_filter_by_wrong_bot_label():
    txt = {
            'input_text': [
                '123\nbot1: ',
                '234\nbot2: ',
                '345\nbot2:butnotlabel'],
            'bot_label': [
                'bot2',
                'bot2',
                'bot2']}

    df = datasets.Dataset.from_dict(txt)
    df = data_cleaning.filter_by_wrong_bot_label(df, col='input_text', bot_label='bot_label')
    assert len(df) == 1
    assert df['input_text'] == ['234\nbot2: ']


def test_clean_by_removing_deleted_messages():
    txt = {
            'input_text': [
                '123\n(deleted): 123\nbot1: ',
                '(deleted): 234\nbot2: ',
                '345\nbot3']}

    df = datasets.Dataset.from_dict(txt)
    df = data_cleaning.clean_by_removing_deleted_messages(df, col='input_text')
    assert df['input_text'] == ['123\nbot1: ', 'bot2: ', '345\nbot3']


def test_clean_by_adding_eos_label():
    txt = {
            'input_text': [
                '123\nuser:</eos>',
                '234\nbot:']}

    df = datasets.Dataset.from_dict(txt)
    df = data_cleaning.clean_by_adding_eos_label(df, col='input_text', eos_label='</eos>')
    assert df['input_text'] == ['123\nuser:</eos>', '234\nbot:</eos>']


def test_format_user_label():
    txt = {
            'input_text': [
                '123\nuser:</eos>',
                '234\nbot:']}

    df = datasets.Dataset.from_dict(txt)
    df = data_cleaning.format_user_label(df, col='input_text', user_label='\nuser:', new_user_label='\nYou:')
    assert df['input_text'] == ['123\nYou:</eos>', '234\nbot:']


def test_construct_pygmalion_prompt():
    context = '123456' * 4
    bot_name = 'Boss'
    context_window = 32
    memory = 'iamboss.' * 2
    prompt = 'hello.'
    res = data_cleaning.construct_pygmalion_prompt(
            context, bot_name, context_window, memory=memory, prompt=prompt)
    assert res == "Boss's Persona: iamboss.iamboss.\n####\n\n<START>\n456123456"
    context_window = 128
    res = data_cleaning.construct_pygmalion_prompt(
            context, bot_name, context_window, memory=memory, prompt=prompt)
    assert res == "Boss's Persona: iamboss.iamboss.\n####\nhello.\n<START>\n123456123456123456123456"
