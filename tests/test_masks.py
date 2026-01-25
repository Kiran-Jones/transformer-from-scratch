import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from masks import create_padding_mask, create_look_ahead_mask


class TestMasks(unittest.TestCase):
    def test_padding_mask_keeps_non_pad(self):
        seq = np.array([[1, 0, 2]])
        mask = create_padding_mask(seq)
        self.assertEqual(mask.shape, (1, 1, 1, 3))
        self.assertTrue(np.array_equal(mask[0, 0, 0], np.array([1, 0, 1])))

    def test_padding_mask_uses_pad_id(self):
        seq = np.array([[5, 7, 5, 1]])
        mask = create_padding_mask(seq, pad_id=5)
        self.assertTrue(np.array_equal(mask[0, 0, 0], np.array([0, 1, 0, 1])))

    def test_look_ahead_mask_lower_triangle(self):
        size = 4
        mask = create_look_ahead_mask(size)
        expected = np.tril(np.ones((size, size)))
        self.assertTrue(np.array_equal(mask, expected))


if __name__ == "__main__":
    unittest.main()
