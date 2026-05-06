"""Multi-seed comparison on the 3-marginal Y toy.

Trains MM+SLERP, MM+Linear, MM+SLERP+Biharmonic on the Y toy (lifted to the
2-simplex / positive orthant of S^2) across several seeds and prints
chained + per-segment MMD^2 tables with mean +/- std.

Usage:
    uv run python scripts/run_toy_y_3marginal_compare.py --seeds 0 1 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_toy_y_3marginal import make_dataset  # noqa: E402
from plot_toy_y_simplex_sphere_ot import raw_to_triangle, triangle_to_simplex  # noqa: E402

from surf.evaluation.generation import generate_euclidean_flow, generate_fisher_flow  # noqa: E402
from surf.evaluation.metrics import fgd, mmd_rbf, swd  # noqa: E402
from surf.geometry.sphere import from_orthant, to_orthant  # noqa: E402
from surf.ot.costs import make_spectral_cost_fn  # noqa: E402
from surf.runtime import setup  # noqa: E402
from surf.training.euclidean_flow_trainer import train_multi_marginal_euclidean_flow  # noqa: E402
from surf.training.flow_trainer import train_multi_marginal_flow  # noqa: E402


TIMES = [0.0, 0.5, 1.0]
METHODS = ["MM+SLERP", "MM+Linear", "MM+SLERP+Biharmonic"]
METRIC_FNS = {
    "MMD^2": mmd_rbf,
    "FGD": fgd,
    "SWD": swd,
}


def build_marginals(n_per_class, sigma, seed):
    m0, m1, m2 = make_dataset(n_per_class, sigma, seed)
    tri = [raw_to_triangle(m) for m in (m0, m1, m2)]
    simp = [triangle_to_simplex(t) for t in tri]
    comp = [torch.from_numpy(p).float() for p in simp]
    sphere = [to_orthant(c) for c in comp]
    return comp, sphere


def split_train_test(stages, train_frac, seed):
    rng = np.random.default_rng(seed + 1000)
    train, test = [], []
    for s in stages:
        n = len(s)
        perm = rng.permutation(n)
        cut = int(train_frac * n)
        train.append(s[perm[:cut]])
        test.append(s[perm[cut:]])
    return train, test


def run_seed(seed, args, rt):
    torch.manual_seed(seed)
    np.random.seed(seed)

    comp, sphere = build_marginals(args.n_per_class, args.sigma, seed)
    comp_train, comp_test = split_train_test(comp, args.train_frac, seed)
    sphere_train, sphere_test = split_train_test(sphere, args.train_frac, seed)

    comp_train = [c.to(rt.device) for c in comp_train]
    sphere_train = [c.to(rt.device) for c in sphere_train]
    D = comp_train[0].shape[1]

    results = {}  # method -> {"chained": (n_hops,), "per_seg": (n_hops,)}

    for name in METHODS:
        print(f"\n[seed={seed}] training {name}...")
        if name == "MM+Linear":
            model, _ = train_multi_marginal_euclidean_flow(
                comp_train, TIMES, D,
                n_iters=args.n_iters, batch_size=args.batch_size,
                lr=args.lr, ot_subsample=args.ot_subsample, label=name,
            )
        elif name == "MM+SLERP":
            model, _ = train_multi_marginal_flow(
                sphere_train, TIMES, D,
                n_iters=args.n_iters, batch_size=args.batch_size,
                lr=args.lr, ot_subsample=args.ot_subsample, label=name,
            )
        elif name == "MM+SLERP+Biharmonic":
            cost_fn = make_spectral_cost_fn(knn=args.biharm_knn, n_eig=args.biharm_neig)
            model, _ = train_multi_marginal_flow(
                sphere_train, TIMES, D,
                n_iters=args.n_iters, batch_size=args.batch_size,
                lr=args.lr, ot_subsample=args.ot_subsample,
                cost_fn=cost_fn, label=name,
            )

        def score(pred_comp, target_comp):
            return {k: float(fn(pred_comp, target_comp)) for k, fn in METRIC_FNS.items()}

        if name == "MM+Linear":
            src_test = comp_test[0].to(rt.device)
            chained_preds, per_seg_preds = [], []
            for i in range(1, len(TIMES)):
                pred = generate_euclidean_flow(
                    model, src_test, n_steps=args.n_gen_steps,
                    t_start=0.0, t_end=TIMES[i],
                )
                chained_preds.append((pred, comp_test[i]))
            for i in range(len(TIMES) - 1):
                pred = generate_euclidean_flow(
                    model, comp_test[i].to(rt.device), n_steps=args.n_gen_steps,
                    t_start=TIMES[i], t_end=TIMES[i + 1],
                )
                per_seg_preds.append((pred, comp_test[i + 1]))
        else:
            src_test = sphere_test[0].to(rt.device)
            chained_preds, per_seg_preds = [], []
            for i in range(1, len(TIMES)):
                pred = generate_fisher_flow(
                    model, src_test, n_steps=args.n_gen_steps,
                    t_start=0.0, t_end=TIMES[i],
                )
                chained_preds.append((from_orthant(pred), comp_test[i]))
            for i in range(len(TIMES) - 1):
                pred = generate_fisher_flow(
                    model, sphere_test[i].to(rt.device), n_steps=args.n_gen_steps,
                    t_start=TIMES[i], t_end=TIMES[i + 1],
                )
                per_seg_preds.append((from_orthant(pred), comp_test[i + 1]))

        results[name] = {"chained": {}, "per_seg": {}}
        for metric in METRIC_FNS:
            results[name]["chained"][metric] = np.array(
                [score(p, t)[metric] for p, t in chained_preds]
            )
            results[name]["per_seg"][metric] = np.array(
                [score(p, t)[metric] for p, t in per_seg_preds]
            )

    return results


def print_table(title, col_names, method_to_seeds):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    w = 18
    header = f"  {'Method':<24}" + "  ".join(f"{c:>{w}}" for c in col_names) + f"  {'mean':>{w}}"
    print(header)
    print("  " + "-" * (24 + (w + 2) * (len(col_names) + 1)))
    for name, seeds in method_to_seeds.items():
        arr = np.stack(seeds)  # (n_seeds, n_hops)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        row_mean = arr.mean(axis=1)
        cols = "  ".join(f"{mean[i]:.4f}\u00b1{std[i]:.4f}".rjust(w) for i in range(len(mean)))
        total = f"{row_mean.mean():.4f}\u00b1{row_mean.std():.4f}".rjust(w)
        print(f"  {name:<24}{cols}  {total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--n-iters", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ot-subsample", type=int, default=512)
    parser.add_argument("--n-gen-steps", type=int, default=50)
    parser.add_argument("--biharm-knn", type=int, default=15)
    parser.add_argument("--biharm-neig", type=int, default=30)
    args = parser.parse_args()

    rt = setup("auto")
    chained_by_method = {metric: {m: [] for m in METHODS} for metric in METRIC_FNS}
    per_seg_by_method = {metric: {m: [] for m in METHODS} for metric in METRIC_FNS}

    for seed in args.seeds:
        print("\n" + "#" * 60)
        print(f"#  SEED {seed}")
        print("#" * 60)
        res = run_seed(seed, args, rt)
        for m in METHODS:
            for metric in METRIC_FNS:
                chained_by_method[metric][m].append(res[m]["chained"][metric])
                per_seg_by_method[metric][m].append(res[m]["per_seg"][metric])

    chained_cols = [f"0->{t}" for t in TIMES[1:]]
    per_seg_cols = [f"{TIMES[i]}->{TIMES[i + 1]}" for i in range(len(TIMES) - 1)]

    n = len(args.seeds)
    for metric in METRIC_FNS:
        print_table(f"CHAINED {metric}  ({n} seeds)", chained_cols, chained_by_method[metric])
        print_table(f"PER-SEGMENT {metric}  ({n} seeds)", per_seg_cols, per_seg_by_method[metric])


if __name__ == "__main__":
    main()
