from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from surf.data.embryoid import load_embryoid_body
from surf.evaluation.metrics import fgd, mmd_rbf, swd
from surf.geometry.sphere import to_compositional


def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto the probability simplex."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")

    u = np.sort(x, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, x.shape[1] + 1, dtype=np.float64)[None, :]
    cond = u - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(x.shape[0]), rho] / (rho + 1)
    w = np.maximum(x - theta[:, None], 0.0)
    return w.astype(np.float32, copy=False)


def resolve_protocol(protocol: str, n_stages: int) -> tuple[list[int], set[int]]:
    if protocol == "full_5stage":
        held_set = set()
    elif protocol == "loo_t2":
        held_set = {2}
    elif protocol == "otpfm_holdout":
        held_set = {i for i in range(n_stages) if i % 2 == 1}
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    train_idx = [i for i in range(n_stages) if i not in held_set]
    return train_idx, held_set


def fit_protocol_pca(train_comp: list[np.ndarray], train_idx: list[int], n_components: int) -> PCA:
    x_train = np.concatenate([train_comp[i] for i in train_idx], axis=0)
    n_components = min(n_components, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(x_train)
    return pca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="embryoid_body.h5ad")
    parser.add_argument("--n-hvg", type=int, default=1500)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["full_5stage", "loo_t2", "otpfm_holdout"],
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--swd-projections", type=int, default=50)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    data = load_embryoid_body(args.input, n_hvg=args.n_hvg, seed=42)
    stages = data["train"]["stages"]
    n_stages = len(stages)
    stage_times = [i / (n_stages - 1) for i in range(n_stages)]

    train_comp = [
        to_compositional(data["train"]["cells"][stage]).cpu().numpy().astype(np.float32, copy=False)
        for stage in stages
    ]
    test_comp = [
        to_compositional(data["test"]["cells"][stage]).cpu().numpy().astype(np.float32, copy=False)
        for stage in stages
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"embryoid_protocol_pca_oracle_hvg{args.n_hvg}.csv"

    rows: list[dict[str, object]] = []
    for protocol in args.protocols:
        train_idx, held_set = resolve_protocol(protocol, n_stages)
        pcas = {
            k: fit_protocol_pca(train_comp, train_idx, k)
            for k in args.dims
        }

        for k, pca in pcas.items():
            cve = float(pca.explained_variance_ratio_.sum())
            for i in range(1, n_stages):
                target = test_comp[i]
                recon = pca.inverse_transform(pca.transform(target))
                recon = project_to_simplex(recon)

                rows.append(
                    {
                        "protocol": protocol,
                        "dims": k,
                        "stage_idx": i,
                        "stage": stages[i],
                        "time": stage_times[i],
                        "is_train_stage": i in train_idx,
                        "is_heldout_stage": i in held_set,
                        "ambient_cve_train": cve,
                        "mmd2": float(mmd_rbf(
                            np_to_torch(recon),
                            np_to_torch(target),
                        )),
                        "fgd": float(fgd(recon, target)),
                        "swd": float(swd(recon, target, n_projections=args.swd_projections)),
                    }
                )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "protocol",
                "dims",
                "stage_idx",
                "stage",
                "time",
                "is_train_stage",
                "is_heldout_stage",
                "ambient_cve_train",
                "mmd2",
                "fgd",
                "swd",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    for protocol in args.protocols:
        print(f"\n[{protocol}]")
        for k in args.dims:
            matching = [r for r in rows if r["protocol"] == protocol and r["dims"] == k]
            cve = matching[0]["ambient_cve_train"]
            print(f"  PCA-{k} train-CVE={cve:.4f}")
            for row in matching:
                print(
                    "   "
                    f"t={row['time']:.2f} {row['stage']}: "
                    f"MMD^2={row['mmd2']:.4f}, FGD={row['fgd']:.4f}, SWD={row['swd']:.4f}, "
                    f"{'TR' if row['is_train_stage'] else 'HO'}"
                )


def np_to_torch(x: np.ndarray):
    import torch

    return torch.from_numpy(x)


if __name__ == "__main__":
    main()
