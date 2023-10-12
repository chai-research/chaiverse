from functools import partial
import numpy as np

from chaiverse.dev import utils
from chaiverse.dev.data_preparation import data_utils


def slice_columns(df, cols):
    cols = utils.ensure_is_list(cols)
    rm_cols = [i for i in df.column_names if i not in cols]
    df = df.remove_columns(rm_cols)
    return df


def rename_columns(df, col, target_col):
    return df.rename_column(col, target_col)


def filter_by_column_value(df, col, value=None):
    ixs = np.array(df[col]) == value
    df = utils.slice_dataset(df, ~ixs)
    return df


def filter_by_ending_with_bot_response(df, col, user_label='Anonymous user: '):
    ixs = np.array([i.split('\n')[-1].startswith(user_label) for i in df[col]])
    df = utils.slice_dataset(df, ~ixs)
    return df


def filter_by_wrong_bot_label(df, col, bot_label='bot_label', n_jobs=1):
    func = partial(_check_last_bot_label, col=col, bot_label=bot_label)
    ixs = data_utils.parallelize_map(df, func=func, n_jobs=n_jobs)
    df = utils.slice_dataset(df, ixs)
    return df


def _check_last_bot_label(row, col, bot_label):
    expected_label = f'{row[bot_label]}: '
    return row[col].split('\n')[-1].startswith(expected_label)


def clean_by_removing_deleted_messages(df, col, n_jobs=1):
    df = data_utils.parallelize_map(
            df,
            func=_remove_if_contains_deleted_message,
            col=col,
            inplace=True,
            n_jobs=n_jobs)
    return df


def _remove_if_contains_deleted_message(txt):
    msg_tag = '(deleted): '
    res = []
    convos = txt.split('\n')
    for txt in convos:
        if msg_tag not in txt:
            res.append(txt)
    return '\n'.join(res)


def clean_by_adding_eos_label(df, col, eos_label='\n', n_jobs=1):
    def _ensure_endswith_eos_label(txt):
        if not txt.endswith(eos_label):
            txt += eos_label
        return txt

    df = data_utils.parallelize_map(
            df,
            func=_ensure_endswith_eos_label,
            col=col,
            inplace=True,
            n_jobs=n_jobs)
    return df


def format_user_label(df, col, user_label, new_user_label, n_jobs=1):
    def _replace_label(txt):
        txt = txt.replace(user_label, new_user_label)
        return txt

    df = data_utils.parallelize_map(
            df,
            func=_replace_label,
            col=col,
            inplace=True,
            n_jobs=n_jobs)
    return df


def format_with_pygmalion_prompt(
        df, col, output_col, bot_label, context_window, memory='memory', prompt='prompt', n_jobs=1):
    def _formatting_in_row(row):
        formatted_txt = _construct_pygmalion_prompt(
                context=row[col],
                bot_name=row[bot_label],
                context_window=context_window,
                memory=row[memory],
                prompt=row[prompt])
        return formatted_txt

    df = data_utils.parallelize_map(
            df,
            func=_formatting_in_row,
            output_col=output_col,
            inplace=True,
            n_jobs=n_jobs)
    return df


def _construct_pygmalion_prompt(context, bot_name, context_window, memory=None, prompt=None):
    text = context[-context_window:]
    trunc_memory = trunc_prompt = ''
    if memory is not None:
        max_memory_window = context_window // 2
        trunc_memory = memory[:max_memory_window]
        trunc_memory = f"{bot_name}'s Persona: {trunc_memory}\n####\n"
    if prompt is not None:
        prompt_window = context_window - len(text) - len(trunc_memory)
        prompt_window = max(prompt_window, 0)
        if prompt_window > 0:
            trunc_prompt = prompt[-prompt_window:]
        trunc_prompt = f"{trunc_prompt}\n<START>\n"
    convos_window = context_window - len(trunc_memory) - len(trunc_prompt)
    trunc_text = text[-convos_window:]
    formatted_text = f"{trunc_memory}{trunc_prompt}{trunc_text}"
    return formatted_text
