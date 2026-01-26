

import numpy as np


def create_padding_mask(seq, pad_id=0):
    # 1 = keep, 0 = mask (pad)
    seq = (seq != pad_id)
    return seq[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    # 1 = keep (lower triangle), 0 = mask (future positions)
    mask = np.tril(np.ones((size, size)))
    return mask
