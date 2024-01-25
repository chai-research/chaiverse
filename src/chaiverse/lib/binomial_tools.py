import numpy as np

def get_ratio(p, q):
    count = p + q
    ratio = p / np.float64(count)
    return ratio


def get_ratio_se(p, q):
    count = p + q
    ratio = get_ratio(p, q)
    ratio_se = ratio * (1 - ratio) / np.float64(count ** 0.5)
    return ratio_se
