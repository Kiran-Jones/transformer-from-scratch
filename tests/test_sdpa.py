import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sdpa import sdpa


class TestSdpa(unittest.TestCase):
    def test_sdpa_mask_blocks_future_positions(self):
        # One query, two keys/values.
        Q = np.array([[[[1.0, 0.0]]]])  # (B=1, H=1, Tq=1, d=2)
        K = np.array([[[[1.0, 0.0], [10.0, 0.0]]]])  # (1, 1, Tk=2, d=2)
        V = np.array([[[[1.0, 2.0], [100.0, 200.0]]]])  # (1, 1, Tk=2, d=2)

        # Keep only the first key/value.
        mask = np.array([[[[1.0, 0.0]]]])

        out = sdpa(Q, K, V, mask=mask)
        self.assertTrue(np.allclose(out[0, 0, 0], np.array([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
