import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch

from surf.geometry.sphere import normalize_sphere, to_orthant
from surf.runtime import setup
from surf.training.flow_trainer import train_multi_marginal_flow


def _load_toy_script():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_toy_y_flow_matching.py"
    spec = importlib.util.spec_from_file_location("run_toy_y_flow_matching", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ToyYSingleBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy = _load_toy_script()
        setup("cpu")

    def test_left_branch_dataset_only_contains_one_target_branch(self):
        source, target = self.toy.make_y_dataset(
            n_source=20,
            n_target_per_branch=12,
            noise=0.0,
            seed=7,
            target_mode="left",
        )
        self.assertEqual(source.simplex.shape, (20, 3))
        self.assertEqual(target.simplex.shape, (12, 3))
        self.assertEqual(set(target.branch.tolist()), {1})

    def test_single_branch_premetric_training_smoke(self):
        source, target = self.toy.make_y_dataset(
            n_source=12,
            n_target_per_branch=12,
            noise=0.0,
            seed=11,
            target_mode="left",
        )
        source_train, _ = self.toy.split_marginal(source, train_frac=0.75, seed=11)
        target_train, _ = self.toy.split_marginal(target, train_frac=0.75, seed=12)

        stage_cells = [
            normalize_sphere(to_orthant(torch.tensor(source_train.simplex, dtype=torch.float32))),
            normalize_sphere(to_orthant(torch.tensor(target_train.simplex, dtype=torch.float32))),
        ]

        model, losses = train_multi_marginal_flow(
            stage_cells=stage_cells,
            stage_times=[0.0, 1.0],
            D=3,
            n_iters=1,
            batch_size=4,
            lr=3e-4,
            label="ToyYSingleBranchSmoke",
            ot_subsample=8,
            premetric_type="biharmonic",
            premetric_ot_cost=True,
            premetric_trajectory_mode="graph_geodesic",
            premetric_knn=5,
            premetric_n_eig=6,
            premetric_extension_k=8,
            premetric_decode_k=8,
        )

        self.assertIsNotNone(model)
        self.assertEqual(len(losses), 1)
        self.assertTrue(math.isfinite(float(losses[0])))


if __name__ == "__main__":
    unittest.main()
