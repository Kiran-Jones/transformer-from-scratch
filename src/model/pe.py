

import numpy as np
from math import sin, cos


def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2):
            PE[pos, 2*i] = sin(pos / 10000 ** (2*i/d_model))
            PE[pos, 2*i+1] = cos(pos / 10000 ** (2*i/d_model))

    return PE

