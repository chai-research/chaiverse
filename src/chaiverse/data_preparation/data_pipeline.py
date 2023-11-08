import logging

from chaiverse.data_preparation import data_utils
from chaiverse import utils


logger = logging.getLogger(__name__)


class DataPipeline():

    def __init__(self, steps):
        self.steps = steps

    def run(self, df, seed=1):
        with data_utils.set_temp_seed(seed):
            for func in self.steps:
                df = self.apply_processor(df, func)
        return df

    def apply_processor(self, df, func):
        utils.check_dataset_format(df)
        df = DataProcessTracker(func)(df)
        return df


class DataProcessTracker():
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, df, return_index=False):
        size_before = len(df)
        processor_name = str(self.processor)
        logger.info('Start processor: %s', processor_name)
        df = self.processor(df)
        logger.info('Data size before: %s, Data size after: %s', size_before, len(df))
        return df
