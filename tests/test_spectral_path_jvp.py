import unittest

import torch

from surf.geometry.spectral_path import SpectralKNNDecoder
from surf.geometry.sphere import normalize_sphere
from surf.runtime import setup
from surf.training.spectral_jvp_trainer import train_spectral_path_jvp_flow


class SpectralPathJVPTest(unittest.TestCase):
    def setUp(self):
        setup("cpu")

    def _stage_cells(self, n=8):
        torch.manual_seed(0)
        a = normalize_sphere(torch.rand(n, 3) + torch.tensor([1.5, 0.2, 0.1]))
        b = normalize_sphere(torch.rand(n, 3) + torch.tensor([0.2, 1.2, 0.6]))
        return [a.float(), b.float()]

    def test_knn_decoder_returns_positive_sphere_points(self):
        cells = torch.cat(self._stage_cells(), dim=0)
        embeddings = torch.randn(len(cells), 2)
        decoder = SpectralKNNDecoder(
            embeddings,
            cells,
            k=3,
            tau="auto",
            chunk_size=4,
            decode_device="cpu",
        )
        decoded = decoder.decode(torch.randn(5, 2))
        self.assertEqual(decoded.shape, (5, 3))
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertGreaterEqual(float(decoded.min()), 0.0)
        torch.testing.assert_close(
            decoded.norm(dim=-1),
            torch.ones(5),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_trainer_smoke(self):
        cells = self._stage_cells()
        model, losses = train_spectral_path_jvp_flow(
            cells,
            [0.0, 1.0],
            D=3,
            n_iters=2,
            batch_size=4,
            lr=1e-3,
            label="TestSpectralJVP",
            ot_subsample=8,
            spectral_knn=3,
            spectral_n_eig=2,
            spectral_weight_power=0.0,
            decode_k=3,
            decode_tau="auto",
            encoder_iters=2,
            encoder_batch_size=4,
            encoder_hidden_dim=16,
            encoder_depth=2,
            diagnostics=False,
        )
        self.assertTrue(losses)
        out = model(cells[0][:2], torch.zeros(2))
        self.assertEqual(out.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
