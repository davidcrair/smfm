import unittest

import numpy as np
import torch

from surf.ot.costs import make_spectral_cost_fn
from surf.training.method_registry import resolve_method_kwargs


class MethodRegistryTest(unittest.TestCase):
    def setUp(self):
        self.y0 = torch.tensor(
            [
                [0.90, 0.08, 0.02],
                [0.65, 0.30, 0.05],
                [0.55, 0.20, 0.25],
            ],
            dtype=torch.float32,
        )
        self.y1 = torch.tensor(
            [
                [0.20, 0.75, 0.05],
                [0.10, 0.30, 0.60],
                [0.05, 0.15, 0.80],
            ],
            dtype=torch.float32,
        )
        self.y0 = self.y0 / self.y0.norm(dim=-1, keepdim=True)
        self.y1 = self.y1 / self.y1.norm(dim=-1, keepdim=True)

    def test_slerp_squared_spectral_uses_configured_spectral_cost(self):
        kwargs = resolve_method_kwargs(
            "MM+SLERP+SquaredSpectral",
            premetric_knn=2,
            premetric_n_eig=2,
            premetric_spectral_family="power",
            premetric_weight_power=0.5,
        )
        expected = make_spectral_cost_fn(
            knn=2,
            n_eig=2,
            spectral_family="power",
            weight_power=0.5,
        )
        np.testing.assert_allclose(
            kwargs["cost_fn"](self.y0, self.y1),
            expected(self.y0, self.y1),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_slerp_biharmonic_ignores_generic_spectral_settings(self):
        kwargs = resolve_method_kwargs(
            "MM+SLERP+Biharmonic",
            premetric_knn=2,
            premetric_n_eig=2,
            premetric_spectral_family="diffusion",
            premetric_weight_power=0.25,
            premetric_diffusion_time=7.0,
        )
        expected = make_spectral_cost_fn(
            knn=2,
            n_eig=2,
            spectral_family="power",
            weight_power=1.0,
        )
        np.testing.assert_allclose(
            kwargs["cost_fn"](self.y0, self.y1),
            expected(self.y0, self.y1),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_slerp_graph_smooth_maps_to_trainer_kwargs(self):
        kwargs = resolve_method_kwargs(
            "MM+SLERP+GraphSmooth",
            graph_smooth_strength=0.02,
            graph_smooth_knn=9,
            graph_smooth_batch_edges=128,
            graph_smooth_sigma_scale=1.5,
        )
        self.assertEqual(
            kwargs,
            {
                "graph_smooth_lambda": 0.02,
                "graph_smooth_knn": 9,
                "graph_smooth_batch_edges": 128,
                "graph_smooth_sigma_scale": 1.5,
            },
        )

    def test_spectral_path_jvp_maps_spectral_settings(self):
        kwargs = resolve_method_kwargs(
            "MM+SpectralPathJVP@alpha=0",
            premetric_knn=9,
            premetric_n_eig=12,
            premetric_spectral_family="power",
            premetric_weight_power=0.5,
            premetric_diffusion_time=2.0,
        )
        self.assertEqual(
            kwargs,
            {
                "spectral_knn": 9,
                "spectral_n_eig": 12,
                "spectral_family": "power",
                "spectral_weight_power": 0.0,
                "spectral_diffusion_time": 2.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
