

import numpy as np
from .utils import softmax

def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask=None, return_weights=False) -> np.ndarray:
    assert type(Q) == np.ndarray
    assert type(K) == np.ndarray
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[-2] == V.shape[-2]

    d_k = Q.shape[-1]

    # compute dot products of query with all keys
    scores = Q @ np.swapaxes(K, -2, -1)

    # divide each by sqrt(d_k)
    scores /= np.sqrt(d_k)

    # apply mask (mask == 0 means masked)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # apply softmax
    attn_weights = softmax(scores)

    # multiply with value matrix
    if return_weights:
        return attn_weights @ V, attn_weights
    return attn_weights @ V

def sdpa_backward(grad_output: np.ndarray, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask=None):
    d_k = Q.shape[-1]

    # Recompute scores
    scores = Q @ np.swapaxes(K, -2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attn_weights = softmax(scores)

    # Gradients w.r.t V
    grad_V = np.matmul(np.swapaxes(attn_weights, -2, -1), grad_output)

    # Gradients w.r.t attention weights
    grad_attn_weights = np.matmul(grad_output, np.swapaxes(V, -2, -1))

    # Gradients w.r.t scores
    grad_scores = attn_weights * (grad_attn_weights - (grad_attn_weights * attn_weights).sum(axis=-1, keepdims=True))

    # Gradients w.r.t Q and K
    grad_Q = np.matmul(grad_scores, K) / np.sqrt(d_k)
    grad_K = np.matmul(np.swapaxes(grad_scores, -2, -1), Q) / np.sqrt(d_k)

    return grad_Q, grad_K, grad_V
