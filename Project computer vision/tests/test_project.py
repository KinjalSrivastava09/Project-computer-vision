import unittest
import numpy as np
import torch
from project import make_dataset, make_image, ShapeCNN, sobel

class ProjectTests(unittest.TestCase):
    def test_dataset_size_balance_and_reproducibility(self):
        x, y = make_dataset(per_class=4, seed=7)
        xx, yy = make_dataset(per_class=4, seed=7)
        self.assertEqual(x.shape, (12, 1, 32, 32))
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(y, yy)
        np.testing.assert_array_equal(np.bincount(y), [4, 4, 4])
        self.assertTrue(np.all((x>=0)&(x<=1)))

    def test_shapes_and_parameter_counts(self):
        for pooling, expected in [(True, 34147), (False, 525667)]:
            model = ShapeCNN(pooling)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(tuple(model(torch.zeros(2, 1, 32, 32)).shape), (2, 3))

    def test_sobel_constant(self):
        gx, gy, magnitude = sobel(np.zeros((32, 32), dtype=np.float32))
        self.assertTrue(np.allclose(gx, 0))
        self.assertTrue(np.allclose(gy, 0))
        self.assertTrue(np.allclose(magnitude, 0))

if __name__ == '__main__':
    unittest.main()
