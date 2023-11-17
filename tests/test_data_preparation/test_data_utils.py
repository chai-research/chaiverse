import pytest
import numpy as np

from chaiverse.data_preparation import data_utils


def test_set_temp_seed():
    state = np.random.get_state()
    with data_utils.set_temp_seed(123):
        res1 = np.random.random()
    with data_utils.set_temp_seed(123):
        res2 = np.random.random()
    with data_utils.set_temp_seed(234):
        res3 = np.random.random()
        tmp_state = np.random.get_state()
    end_state = np.random.get_state()
    assert res1 == res2
    assert res2 != res3
    assert (state[1] == end_state[1]).all()
    assert (tmp_state[1] != end_state[1]).any()


def _func1(x):
    return x + '1'


def _row_wise_func2(x):
    return x['input_text'] + str(x['labels'])


def test_parallelize_map_with_datasetdict_as_input(hf_dataset):
    res = data_utils.parallelize_map(hf_dataset, func=_func1, col='input_text')
    assert all(res['train'] == np.array(['a1', 'b1', 'c1', 'd1', 'e1', 'f1']))
    assert all(res['validation'] == np.array(['o1', 'p1', 'q1', 'r1']))

    res = data_utils.parallelize_map(hf_dataset, func=_func1, inplace=True, col='input_text')
    assert all(res['train']['input_text'] == np.array(['a1', 'b1', 'c1', 'd1', 'e1', 'f1']))


def test_parallelize_map_with_dataset_as_input(hf_dataset):
    res = data_utils.parallelize_map(hf_dataset['train'], func=_func1, col='input_text')
    assert all(res == np.array(['a1', 'b1', 'c1', 'd1', 'e1', 'f1']))

    res = data_utils.parallelize_map(hf_dataset['train'], func=_func1, inplace=True, col='input_text')
    assert all(res['input_text'] == np.array(['a1', 'b1', 'c1', 'd1', 'e1', 'f1']))


def test_parallelize_map_not_change_input(hf_dataset):
    expected_input = np.array(hf_dataset['train']['input_text'])
    df = data_utils.parallelize_map(hf_dataset, func=_func1, inplace=True, col='input_text')
    assert df != hf_dataset
    assert all(np.array(hf_dataset['train']['input_text']) == expected_input)


def test_parallelize_map_with_njobs(hf_dataset):
    res1 = data_utils.parallelize_map(hf_dataset['train'], func=_func1, col='input_text', n_jobs=1)
    res2 = data_utils.parallelize_map(hf_dataset['train'], func=_func1, col='input_text', n_jobs=-1)
    assert (res1 == res2).all()


def test_parallelize_map_with_row_wise_function(hf_dataset):
    res = data_utils.parallelize_map(hf_dataset, func=_row_wise_func2)
    input1 = hf_dataset['train']['input_text']
    input2 = hf_dataset['train']['labels']
    expected_output = np.array([str(i) + str(j) for i, j in zip(input1, input2)])
    assert all(res['train'] == expected_output)


def test_parallelize_map_in_inplace_mode(hf_dataset):
    res = data_utils.parallelize_map(hf_dataset['train'], func=_func1, col='input_text', inplace=True)
    assert res.shape == hf_dataset['train'].shape
    assert all(res['input_text'] == np.array(['a1', 'b1', 'c1', 'd1', 'e1', 'f1']))

    res = data_utils.parallelize_map(
            hf_dataset['train'], func=_func1, col='input_text', output_col='test123', inplace=True)
    assert res.shape[1] == hf_dataset['train'].shape[1] + 1
    assert all(res['test123'] == np.array(['a1', 'b1', 'c1', 'd1', 'e1', 'f1']))
    assert all(res['input_text'] == np.array(hf_dataset['train']['input_text']))


def test_parallelize_map_in_inplace_mode_with_output_column(hf_dataset):
    res = data_utils.parallelize_map(hf_dataset, func=_row_wise_func2, inplace=True, output_col='test123')
    input1 = hf_dataset['train']['input_text']
    input2 = hf_dataset['train']['labels']
    expected_output = np.array([str(i) + str(j) for i, j in zip(input1, input2)])
    assert all(res['train']['test123'] == expected_output)


def test_parallelize_map_inplace_model_must_has_output_column(hf_dataset):
    with pytest.raises(AssertionError) as e:
        data_utils.parallelize_map(hf_dataset, func=_row_wise_func2, inplace=True)
    assert 'column in inplace mode' in str(e)
