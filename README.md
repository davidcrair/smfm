# Data-faithful Fisher Flow Matching

CPSC 5860 final project. Adds a data-manifold-aware interpolant to Fisher
Flow Matching on the positive orthant of the hypersphere.

## Motivation

Vanilla Fisher Flow Matching (Davis et al., NeurIPS 2024) interpolates
coupled source/target cells along a great-circle SLERP, which traces the
shortest arc on the sphere. For scRNA-seq data this arc cuts through
empty regions of the data manifold because the biological "ribbon" is a
thin, curved subset of the orthant. We replace the SLERP interpolant
with a piecewise-SLERP polyline through real training cells (shortest
path on a kNN graph weighted by angular distance) and make the OT
coupling cost match by also using graph shortest-path distance, so
coupling and interpolant live in the same geometry.

## Ablation grid

|                          | Sphere OT cost            | Graph OT cost              |
| ------------------------ | ------------------------- | -------------------------- |
| **SLERP interpolant**    | Vanilla Fisher Flow       | Cost-only swap (inconsistent) |
| **Polyline interpolant** | Path-only swap (inconsistent) | **Data-faithful (consistent)** |

## Evaluation

Leave-one-timepoint-out on embryoid body. Train flows between stages on
either side of a held-out middle stage, integrate to `t=0.5`, score
MMD² against the held-out intermediate stage cells. Endpoint-only
metrics can't distinguish polyline from SLERP, because both can hit the
same endpoint — the difference only shows up in intermediate states.

## Running

Place `embryoid_body.h5ad` in this directory, then:

```bash
uv sync
uv run python main.py embryoid
# knobs: --n-cells, --ot-subsample, --n-iters, --batch-size, --knn
```

For a fast sanity check on synthetic data:

```bash
uv run python main.py toy
```

## Outputs

- `embryoid_results.html` — PHATE embedding, rows = LOO transitions,
  columns = ground truth + 4 ablation methods. Diamonds are `t=0.5`
  predictions; they should land on the held-out stage.
- `embryoid_loss_curves.html` — training loss per (transition, method).
- `ot_coupling_<src>_<tgt>.png` — side-by-side sphere vs graph OT
  couplings on the first LOO transition.
