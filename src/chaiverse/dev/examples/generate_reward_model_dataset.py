import logging
from functools import partial

from chaiverse.dev.data_preparation import data_cleaning as dc
from chaiverse.dev.data_preparation import reward_data_cleaning as rdc
from chaiverse.dev.data_preparation.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

N_JOBS = 10
FEEDBACK_DATA = 'ChaiML/20231007_chai_prize_model_feedback_all'
BOT_CONFIG_DATA = 'ChaiML/seasonIII_chatAI_configurations'

STEPS = [
        partial(dc.filter_by_column_value, col='bot_label', value=None),
        partial(dc.clean_by_removing_deleted_messages, col='conversation', n_jobs=N_JOBS),
        partial(dc.filter_by_ending_with_bot_response, col='conversation'),
        partial(dc.filter_by_wrong_bot_label, col='conversation', bot_label='bot_label', n_jobs=N_JOBS),
        partial(dc.clean_by_adding_eos_label, col='conversation', eos_label='\n', n_jobs=N_JOBS),
        partial(dc.format_user_label, col='conversation', user_label='Anonymous user: ', new_user_label='You: ', n_jobs=N_JOBS),
        partial(dc.format_with_pygmalion_prompt, col='conversation', bot_label='bot_label', output_col='input_text', context_window=2048, n_jobs=N_JOBS),
        rdc.format_thumbs_up_labels,
        partial(dc.slice_columns, cols=['input_text', 'labels']),
        ]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = rdc.load_feedback_dataset_with_bot_info(
            feedback_path=FEEDBACK_DATA,
            bot_path=BOT_CONFIG_DATA,
            )
    data_pipeline = DataPipeline(steps=STEPS)
    data = data_pipeline.run(df)
