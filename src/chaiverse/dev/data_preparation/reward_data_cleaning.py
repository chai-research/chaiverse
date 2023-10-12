from datasets import Dataset, concatenate_datasets
import numpy as np

from chaiverse.dev.utils import load_dataset


def load_feedback_dataset_with_bot_info(feedback_path, bot_path):
    df = load_dataset(feedback_path)
    bot_df = load_dataset(bot_path)
    df = merge_with_bot_df(df, bot_df)
    return df


def merge_with_bot_df(df, bot_df):
    bot_dict = convert_bot_df_to_dict(bot_df)
    bot_info = []
    for row in df:
        bot_id = row['bot_id']
        bot_info.append(bot_dict.get(bot_id, {}))
    expanded_bot_df = Dataset.from_list(bot_info)
    df = concatenate_datasets([df, expanded_bot_df], axis=1)
    return df


def convert_bot_df_to_dict(bot_df):
    res = {}
    for row in bot_df:
        bot_id = row.pop('bot_id')
        res[bot_id] = row
    return res


def format_thumbs_up_labels(df):
    labels = np.array(df['thumbs_up']).astype(int)
    df = df.add_column('labels', labels)
    return df
