import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from embedding import Embedding


class TestEmbeddingGrad(unittest.TestCase):
    def test_embedding_backward_accumulates_grads(self):
        emb = Embedding(vocab_size=5, d_model=3)

        ids1 = np.array([[1, 2]])
        ids2 = np.array([[2, 3]])

        out1 = emb.forward(ids1)
        emb.backward(np.ones_like(out1), token_ids=ids1)

        out2 = emb.forward(ids2)
        emb.backward(2 * np.ones_like(out2), token_ids=ids2)

        # Token 1 appears once with grad 1.
        self.assertTrue(np.allclose(emb.grad_weights[1], np.ones(3)))
        # Token 2 appears twice: grad 1 + grad 2 = 3.
        self.assertTrue(np.allclose(emb.grad_weights[2], np.ones(3) * 3))
        # Token 3 appears once with grad 2.
        self.assertTrue(np.allclose(emb.grad_weights[3], np.ones(3) * 2))


if __name__ == "__main__":
    unittest.main()
