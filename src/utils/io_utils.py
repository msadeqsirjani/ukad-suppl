import numpy as np
from scipy.io import loadmat


def decode_raw(read_path, dtype, shape, **kwargs):
    x = np.fromfile(read_path, dtype=dtype, **kwargs)
    x = np.reshape(x, shape, order="C")
    return x

def decode_mat(read_path, key, **kwargs):
    x = loadmat(read_path, **kwargs)[key]
    x = np.asarray(x)
    return x
