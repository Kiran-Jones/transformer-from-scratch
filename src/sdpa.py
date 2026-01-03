

import numpy as np
from utils import softmax

def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask=None) -> np.ndarray:
    assert type(Q) == np.ndarray
    assert type(K) == np.ndarray
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[-2] == V.shape[-2]

    d_k = Q.shape[-1]
    
    # compute dot products of query with all keys
    scores = Q @ np.swapaxes(K, -2, -1)

    # divide each by sqrt(d_k)
    scores /= np.sqrt(d_k)

    # apply mask 
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # apply softmax
    attn_weights = softmax(scores)

    # multiply with value matrix
    return attn_weights @ V
