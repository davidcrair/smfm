# Wire GoM and Beijing Through the Multi-Marginal Flow-Matching Pipeline

## Confirmed ground costs in current methods

For the writeup table caption and method section:

| Method (display name)      | Internal name | Ground cost | Notes |
|----------------------------|---------------|-------------|-------|
| Linear FM (sphere)         | `MM+Linear`   | Squared Euclidean on sphere-encoded log1p HVG | `compute_euclidean_cost_matrix` |
| Linear FM (PCA-50)         | `MM+Linear` w/ `space=pca50` | Squared Euclidean on 50-PC latent | same cost_fn, different latent |
| FisherFlow                 | `MM+SLERP`    | Squared sphere arc-length $d^2 = \arccos^2(\langle x,y\rangle)$ | `compute_sphere_cost_matrix` (Davis 2024 Prop. 2 W2 cost) |
| \ourmodel, $\alpha=...$    | `MM+SLERP+SquaredSpectral@alpha=...` | Spectral $1/\lambda^\alpha$ on kNN-Laplacian eigenmaps | `make_spectral_cost_fn` |
| **Linear FM + Random**     | `MM+Linear+Random` | Uniform-random costs (no pairing structure) | New; `compute_random_cost_matrix` |
| **FisherFlow + Random**    | `MM+SLERP+Random`  | Uniform-random costs           | New; same |

Both random-baseline rows are now registered (`surf/training/method_registry.py`)
and produce sensible numbers: on pancreas split=42, MM+Linear+Random gives
chained MMD$^2$ mean 0.21 vs MM+Linear's 0.17 and the do-nothing baseline's
0.42, showing OT pairing buys ~25% improvement over random pairing on top of
what flow matching alone gets you.

**Goal.** Run the existing spectral-OT methods (with optional graph
augmentation) on the Gulf-of-Mexico vortex dataset and the Beijing PM2.5
dataset so we can demonstrate the *connectivity-conditioned* ablation:

- On GoM/Beijing with the bare spectral cost (`aug=0`), trajectories should
  collapse to random pairing because the kNN union graph is disconnected.
- With `aug>0`, the cost matrix recovers a sensible OT plan and the flow
  matches the per-segment marginals.

This is the experimental backbone of the "graceful degradation" theory section
in the paper.

## Why it doesn't work today

`surf/train.py` currently assumes every dataset goes through the
`log1p HVG → compositional → sqrt → unit sphere` pipeline (Fisher-Rao
geometry). That pipeline is meaningful for scRNA-seq counts but not for:

- **GoM**: 2D vortex positions, not compositional. The sphere encoding turns
  arbitrary positions into degenerate orthant points.
- **Beijing PM2.5**: 1D scalar concentrations. `to_compositional` of a 1D
  vector is identically `[1.0]`, so the sphere encoding collapses every point
  to the same location.

Eval also assumes `test_stage_comp` exists and uses MMD/FGD/SWD on
compositional vectors. For raw 2D/1D data, those metrics belong in the raw
coordinate space.

## Design

Add a `data.representation` field to the data configs with two values:

| Value | When | Train rep | Eval rep |
|---|---|---|---|
| `sphere` (default) | EB, pancreas | log1p → compositional → orthant → unit sphere | compositional |
| `raw_euclidean` | GoM, Beijing | identity (use raw coords) | raw coords |

In `surf/train.py`, branch on `cfg.data.representation`:

- `sphere`: existing code path. No changes.
- `raw_euclidean`:
  - `train_stage_cells_sub` = raw tensors (no transformation)
  - `test_stage_comp` is *replaced* with `test_stage_raw` and passed to the
    metric functions as the target distribution.
  - Only Euclidean / spectral methods are valid (skip MM+SLERP, MM+Score_*,
    MM+SI, anything that depends on sphere geometry). Validate this at startup.
  - Skip `to_compositional` / `to_orthant` calls.
  - Skip the stage-width / do-nothing baseline that uses `to_compositional`.

The Euclidean flow trainer (`train_multi_marginal_euclidean_flow`) already
operates on arbitrary R^D inputs, so once the data flow is right the trainer
side needs no changes.

## Tasks

1. **Loaders that match the standard dict shape.**
   - `surf/data/gom.py:load_gom()` returns a dict with `train`/`val`/`test`
     splits, each with `stages` (timepoints `t1`..`t9`), `cells` (mapping
     stage → 2D `Tensor`), `transitions`, `cell_types` set to None or stage
     labels. Calls `download_gom_data` from `~/Downloads/data.py` and applies
     the StandardScaler. **Stratify** train/val/test per stage just like
     embryoid; `seed` controlled by `cfg.data.seed`.
   - `surf/data/beijing.py:load_beijing()` analogous, returning the 13 monthly
     PM2.5 marginals from `data-2.py`. 1D feature vectors, station=Dingling
     (or configurable).

2. **Configs.**
   - `conf/data/gom.yaml`: `name: gom`, `representation: raw_euclidean`,
     `path: data/GoMvortex_data.npy`, `seed: 42`, no `n_hvg`.
   - `conf/data/beijing.yaml`: same shape, `path: data/beijing/...`,
     `station: Dingling`, etc.
   - Add `representation: sphere` field to `conf/data/embryoid.yaml` and
     `conf/data/pancreas.yaml` (explicit default; backward-compatible).

3. **Dispatch in `surf/train.py`.**
   - Extend the loader dispatch to include `gom` and `beijing`.
   - Read `cfg.data.representation` and branch the encoding/eval pipeline.
   - When `raw_euclidean`:
     - Skip the sphere encoding block.
     - Build `train_stage_raw`/`test_stage_raw` directly from the loader
       output.
     - Pass `train_stage_raw` to `train_multi_marginal_euclidean_flow`.
     - Replace `test_stage_comp` with `test_stage_raw` in eval calls.
     - Skip the `Stage-width reference` and `Do-nothing baseline` blocks
       (they assume compositional).
     - Validate that selected methods are in
       `EUCLIDEAN_METHOD_NAMES ∪ {"MM+SLERP+SquaredSpectral"}`. Reject sphere
       methods with a clear error.

4. **Methods to support on `raw_euclidean`.**
   - `MM+Linear` — Euclidean flow, Euclidean OT cost.
   - `MM+Linear+Biharmonic` — Euclidean flow, biharmonic spectral OT cost.
   - `MM+Linear+SquaredSpectral` — register a new alias of `MM+Linear` that
     accepts `@alpha=...,aug=...`. Same plumbing as `MM+SLERP+SquaredSpectral`
     in the registry but Euclidean representation.
   - Random-pairing baseline: add `MM+Linear+Random` that wires
     `cost_fn=None` (or a constant cost) so OT degenerates to uniform
     coupling. This is the **honest baseline** for the "spectral collapses to
     random when disconnected" claim.

5. **Experiment configs.**
   - `conf/experiment/gom_aug_sweep.yaml`:
     ```
     methods:
       - MM+Linear+Random
       - MM+Linear
       - "MM+Linear+SquaredSpectral@alpha=2"
       - "MM+Linear+SquaredSpectral@alpha=2,aug=0.01"
       - "MM+Linear+SquaredSpectral@alpha=2,aug=0.05"
       - "MM+Linear+SquaredSpectral@alpha=2,aug=0.2"
     eval:
       n_seeds: 1
     ```
   - Identical for `conf/experiment/beijing_aug_sweep.yaml`.

6. **Runner scripts.**
   - `scripts/run_gom_aug_kfold.sh`: 5 splits × the 6-method sweep.
   - `scripts/run_beijing_aug_kfold.sh`: same.

7. **Aggregator.**
   - Generalize `scripts/build_pancreas_table.py` to read GoM/Beijing logs.
   - Or write a sister `scripts/build_aug_ablation_table.py` that emits a
     small 6-method × 4-hop table per dataset.
   - Skip the FisherFlow Δ baseline (no FisherFlow row); use Linear+Random as
     the Δ reference instead, since that's the meaningful "do nothing"
     for raw Euclidean settings.

8. **Sanity checks during implementation.**
   - Run `MM+Linear` on GoM with 1 seed, 200 iters; verify the loss decays
     and the do-nothing baseline (re-implemented for raw Euclidean) is small.
   - Run `MM+Linear+SquaredSpectral@alpha=2` (no aug) on GoM and confirm the
     OT cost printout shows `spectral_power_1_cost` and that cost values
     between cross-component pairs are dominated by the indicator term
     (numerically large, comparable across all such pairs).
   - Run `aug=0.05` and confirm the cross-component cost spread *increases*
     (different pairs now have different costs).

## Expected outcomes

- **GoM**:
  - `MM+Linear+Random`: high MMD across all hops (the floor).
  - `MM+Linear`: lower MMD; Euclidean OT works because GoM is 2D and
    Euclidean is fine there.
  - `MM+Linear+SquaredSpectral@alpha=2,aug=0`: ≈ random baseline (the
    degeneracy story).
  - `MM+Linear+SquaredSpectral@alpha=2,aug>0`: between random and Euclidean,
    monotone improvement with `aug`.

- **Beijing**: same shape but more extreme — the 1D fragmentation means the
  spectral cost without augmentation collapses to nearly random pairing in
  every bucket. Augmentation should recover essentially Euclidean-OT-like
  performance.

If those qualitative patterns hold, the augmentation section becomes a
credible "graceful degradation under boundary conditions" story rather than
a hand-wave.

## Estimated effort

- Loaders + configs: 1 hr
- `train.py` branch for `raw_euclidean`: 1.5 hr (the main risk; eval
  pipeline assumes compositional in several places — `eval_chained_metrics`
  and `eval_per_segment_metrics` need a `representation` flag too if not
  already plumbed)
- New method registrations: 30 min
- Experiment configs + runner scripts: 30 min
- Smoke + 5-split sweep: 1 hr (depending on dataset size; both small)
- Aggregator + table: 45 min
- **Total: ~5 hr** for a working ablation table.

## Out of scope

- Connecting the sphere methods (MM+SLERP, MM+SI, MM+Score_*) to GoM/Beijing.
  These don't have a coherent interpretation outside compositional data and
  shouldn't be force-fit.
- Multi-marginal Schrödinger bridge variants (entropic OT, score term). The
  point of this experiment is the cost-side fix; noise scheduling is
  orthogonal.
- Beijing per-station conditioning (the MMFM C-MMFM treatment). Single
  station Dingling is sufficient for the ablation.

## Validation criteria

The implementation is correct if:

1. `MM+Linear` on EB matches the existing baseline number to ±std.
2. `MM+Linear+SquaredSpectral@alpha=2,aug=0` on GoM is *worse* than
   `MM+Linear` (because spectral degenerates).
3. `MM+Linear+SquaredSpectral@alpha=2,aug=0.1` on GoM is *better* than
   `aug=0` (augmentation rescues).
4. The same monotone pattern holds on Beijing.

If (2) does not hold (i.e., spectral without augmentation already works on
GoM), revisit the connectivity diagnostic — something about the cost
computation may be smoothing across components in a way the diagnostic
didn't catch.
