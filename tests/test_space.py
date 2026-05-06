import unittest
from types import SimpleNamespace

import torch

from surf.space import build_space


class SpaceTest(unittest.TestCase):
    def setUp(self):
        coeffs_a = torch.tensor(
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            dtype=torch.float32,
        )
        coeffs_b = torch.tensor(
            [[1.0], [0.0], [2.0], [1.0], [3.0]],
            dtype=torch.float32,
        )
        basis1 = torch.tensor([[1.0, 0.0, 1.0, 2.0]], dtype=torch.float32)
        basis2 = torch.tensor([[0.0, 1.0, 1.0, -1.0]], dtype=torch.float32)
        stage0 = coeffs_a * basis1 + coeffs_b * basis2
        stage1 = (coeffs_a + 1.0) * basis1 + (coeffs_b - 0.5) * basis2
        self.train_cells = [stage0, stage1]

    def test_log1p_space_is_identity(self):
        cfg = SimpleNamespace(name="log1p", n_components=None, random_state=42)
        space = build_space(cfg, self.train_cells)
        encoded = space.encode(self.train_cells[0])
        decoded = space.decode(encoded)
        self.assertEqual(space.output_dim, self.train_cells[0].shape[1])
        self.assertTrue(torch.equal(encoded, self.train_cells[0]))
        self.assertTrue(torch.equal(decoded, self.train_cells[0]))

    def test_pca_space_round_trips_rank_two_data(self):
        cfg = SimpleNamespace(name="pca", n_components=2, random_state=42)
        space = build_space(cfg, self.train_cells)
        encoded = space.encode(self.train_cells[0])
        decoded = space.decode(encoded)
        self.assertEqual(encoded.shape[1], 2)
        self.assertEqual(space.output_dim, 2)
        torch.testing.assert_close(decoded, self.train_cells[0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
