from numpy import isnan

from chaiverse.lib import binomial_tools

def test_get_ratio():
    assert binomial_tools.get_ratio(1, 2) == 1/3
    assert binomial_tools.get_ratio(0, 1) == 0
    assert isnan(binomial_tools.get_ratio(0, 0))


def test_get_ratio_se():
    assert isnan(binomial_tools.get_ratio_se(0, 0))
    assert binomial_tools.get_ratio_se(0, 1) == 0
    assert binomial_tools.get_ratio_se(0, 2) == 0
    expected_se = (1/3) * (2/3) / (3 ** 0.5)
    assert binomial_tools.get_ratio_se(1, 2) - expected_se < 1e-5
