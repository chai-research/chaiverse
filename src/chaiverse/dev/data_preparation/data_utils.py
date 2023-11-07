import contextlib
import multiprocessing as mp

import numpy as np


@contextlib.contextmanager
def set_temp_seed(seed):
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            np.random.set_state(state)


def parallelize_map(df, func, col=None, output_col=None, inplace=False, n_jobs=1):
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    if inplace:
        output_col = output_col or col
        assert output_col is not None, 'must assign output column in inplace mode.'
    else:
        output_col = output_col or '__tmp__'

    if col is None:
        df = df.map(lambda x: {output_col: func(x)}, num_proc=n_jobs)
    else:
        df = df.map(lambda x: {output_col: func(x[col])}, num_proc=n_jobs)
    if inplace:
        return df
    if type(df).__name__ == 'DatasetDict':
        res = {fold: np.array(_df[output_col]) for fold, _df in df.items()}
    else:
        res = np.array(df[output_col])
    return res
