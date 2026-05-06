"""
Data-faithful Fisher Flow Matching on the positive orthant.

Motivation: vanilla Fisher Flow Matching uses a SLERP interpolant between
coupled source/target cells, which traces a great-circle on the sphere.
On biological data the great-circle cuts through empty regions of the data
manifold. Replacing SLERP with a piecewise-SLERP polyline through real cells
(graph shortest path on a kNN graph, parameterized by cumulative arc length)
forces training trajectories to hug the data ribbon, and the corresponding
OT cost becomes the graph shortest-path distance so coupling and interpolant
live in the same geometry.

Ablation grid:
  1. Sphere OT cost  + SLERP interpolant   (vanilla Fisher Flow, baseline)
  2. Graph OT cost   + SLERP interpolant   (cost-only swap, inconsistent)
  3. Sphere OT cost  + Polyline interpolant (path-only swap, inconsistent)
  4. Graph OT cost   + Polyline interpolant (data-faithful, consistent)

Evaluation: leave-one-timepoint-out on embryoid body. Train flows on
(t0 -> t2) and (t2 -> t4), integrate to t=0.5, score MMD against held-out
intermediate stages t1 and t3.

Install: pip install torch pot torchdiffeq scikit-learn scanpy anndata matplotlib plotly phate
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import warnings
import subprocess
import datetime
import sys

warnings.filterwarnings("ignore")


def _print_run_provenance():
    """Log git commit, timestamp, and CLI args at the start of every run.

    Helps diagnose which code version produced which results file, so we
    don't conflate runs across commits (e.g. EMD vs Sinkhorn, old vs new
    score net, etc.).
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        short = commit[:10]
    except Exception:
        commit = "<not a git repo>"
        short = "unknown"
    try:
        subject = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        subject = ""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = " (DIRTY: uncommitted changes)" if status else ""
    except Exception:
        dirty = ""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print("=" * 78)
    print(f"RUN PROVENANCE")
    print(f"  timestamp : {ts}")
    print(f"  commit    : {short}{dirty}")
    if subject:
        print(f"  subject   : {subject}")
    print(f"  command   : {' '.join(sys.argv)}")
    print("=" * 78)


_print_run_provenance()


# ── Device selection ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Performance: enable TF32 matmul on Ampere+ for any fp32 paths that remain,
# use bf16 autocast for training forward/backward (params stay fp32).
USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16 if USE_AMP else torch.float32
if USE_AMP:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── optional deps ────────────────────────────────────────────────────────────
try:
    import phate

    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False
    print("phate not installed — PHATE is only used for visualization; PCA fallback will be used")

try:
    import ot

    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("pot not installed — using uniform coupling (no OT)")

try:
    from torchdiffeq import odeint

    HAS_ODEINT = True
except ImportError:
    HAS_ODEINT = False
    print("torchdiffeq not installed — using Euler integration")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def _get_shared_embedding_matrix(D, embed_seed=0):
    """Return a fixed random projection matrix shared across all data splits."""
    rng_w = np.random.default_rng(embed_seed)
    return rng_w.normal(0, 1, (2, D)).astype(np.float32)


# Pre-compute once so train and test live in the same space
_EMBED_W = None


def make_compositional_data(n=400, D=32, seed=42):
    """
    Simulate control and perturbed cells as compositional (simplex) data.
    Control: points near one region of the manifold.
    Perturbed: points shifted to another region.
    We embed a 2D swiss-roll into D dimensions then softmax to get proportions.

    IMPORTANT: both control and perturbed use the SAME projection matrix W
    so that the perturbation structure is preserved after embedding.
    The same W is also reused across train/test splits.
    """
    global _EMBED_W
    if _EMBED_W is None or _EMBED_W.shape[1] != D:
        _EMBED_W = _get_shared_embedding_matrix(D)

    rng = np.random.default_rng(seed)

    # Control cells: one arm of the swiss roll
    X_ctrl_2d, _ = make_swiss_roll(n_samples=n, noise=0.3, random_state=seed)
    X_ctrl_2d = X_ctrl_2d[:, [0, 2]] / 10.0  # use x,z coords

    # Perturbed cells: shifted version
    X_pert_2d = X_ctrl_2d + np.array([2.5, 1.5])
    X_pert_2d += rng.normal(0, 0.2, X_pert_2d.shape)

    def embed_and_compose(X_2d, D, rng, W):
        # Project through SHARED embedding matrix
        X_D = X_2d @ W + rng.normal(0, 0.1, (len(X_2d), D))
        # Softmax to get valid compositional data (sums to 1, all positive)
        X_D = np.exp(X_D - X_D.max(axis=1, keepdims=True))
        X_D = X_D / X_D.sum(axis=1, keepdims=True)
        return X_D.astype(np.float32)

    X_ctrl = embed_and_compose(X_ctrl_2d, D, rng, _EMBED_W)
    X_pert = embed_and_compose(X_pert_2d, D, rng, _EMBED_W)

    return torch.tensor(X_ctrl), torch.tensor(X_pert)


# ─────────────────────────────────────────────────────────────────────────────
# 2. POSITIVE ORTHANT MAPPING
# ─────────────────────────────────────────────────────────────────────────────


def to_orthant(p):
    """Map compositional data p (simplex) -> sqrt(p) (positive orthant of S^d)."""
    return torch.sqrt(p.clamp(min=1e-8))


def from_orthant(y):
    """Map positive orthant y -> y^2 (back to simplex). y is already unit-norm."""
    return y ** 2


def normalize_sphere(y):
    """Project onto unit sphere."""
    return y / y.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def to_compositional(X):
    """
    Convert log1p-normalized expression to a probability distribution over genes.
    X: (n, D) tensor of log1p(library-size-normalized counts).
    """
    counts = torch.expm1(X).clamp(min=0)
    counts = counts + 1e-8  # avoid exact zeros in downstream sqrt
    return counts / counts.sum(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SLERP AND SPHERE GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTERPOLANTS
# ─────────────────────────────────────────────────────────────────────────────


class SLERPInterpolant:
    """
    Great-circle SLERP interpolant between coupled source/target cells.
    This is the vanilla Fisher Flow Matching interpolant — a single geodesic
    arc on the sphere. Fast, closed-form, but ignores the data manifold.
    """

    def __init__(self, Y0, Y1):
        self.Y0 = Y0
        self.Y1 = Y1
        self.n0 = len(Y0)
        self.n1 = len(Y1)
        self.device = Y0.device

    def sample(self, src_idx, tgt_idx, t):
        """
        src_idx, tgt_idx: (B,) numpy int arrays (local indices into Y0, Y1)
        t: (B, 1) torch tensor on device
        Returns: z_t (B, D), v_t (B, D) on device
        """
        y0 = self.Y0[src_idx]
        y1 = self.Y1[tgt_idx]

        cos_omega = (y0 * y1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)

        w0 = torch.sin((1 - t) * omega) / sin_omega
        w1 = torch.sin(t * omega) / sin_omega
        z_t = w0 * y0 + w1 * y1
        z_t = z_t / z_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # d/dt SLERP = omega * [-cos((1-t)ω)/sin(ω) * y0 + cos(tω)/sin(ω) * y1]
        v_t = omega * (
            -torch.cos((1 - t) * omega) / sin_omega * y0
            + torch.cos(t * omega) / sin_omega * y1
        )
        # Project to tangent space at z_t (handles numerical drift)
        v_t = v_t - (v_t * z_t).sum(dim=-1, keepdim=True) * z_t

        return z_t, v_t


class PolylineGeodesic:
    """
    Data-faithful polyline interpolant on the positive orthant.

    Builds a kNN graph over the union of source and target cells, runs
    Dijkstra from every source, and uses the shortest-path polyline between
    coupled pairs as the training trajectory. The polyline is parameterized
    by cumulative arc length so t∈[0,1] is constant angular speed along the
    whole path.

    Also exposes `cost_matrix` — the graph shortest-path distance from every
    source to every target, which is the consistent OT cost for this
    interpolant. Disconnected pairs fall back to the direct angular distance
    and a two-node degenerate path, equivalent to SLERP for those pairs.
    """

    def __init__(self, Y0, Y1, k=15):
        self.device = Y0.device
        self.Y0 = Y0
        self.Y1 = Y1
        self.n0 = len(Y0)
        self.n1 = len(Y1)
        self.all_nodes = torch.cat([Y0, Y1], dim=0)  # (N, D) on device

        self._build_graph(k)

    def _build_graph(self, k):
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra, connected_components

        A = self.all_nodes.detach().cpu().numpy()
        N = len(A)
        k_eff = min(k + 1, N)

        # Cosine metric on unit vectors → distance = 1 - cos_sim
        nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
        nn.fit(A)
        dists, inds = nn.kneighbors(A)
        cos_sim = np.clip(1.0 - dists, -1 + 1e-6, 1 - 1e-6)
        ang_dists = np.arccos(cos_sim)  # (N, k_eff)

        # Build sparse graph, excluding self-edges (first neighbor is self)
        rows = np.repeat(np.arange(N), k_eff - 1)
        cols = inds[:, 1:].reshape(-1)
        data = ang_dists[:, 1:].reshape(-1)
        graph = csr_matrix((data, (rows, cols)), shape=(N, N))
        graph = graph.maximum(graph.T)  # symmetrize

        # Report connectivity
        n_comp, comp_labels = connected_components(graph, directed=False)
        if n_comp > 1:
            sizes = np.bincount(comp_labels)
            print(f"    kNN graph has {n_comp} connected components (largest={sizes.max()}, smallest={sizes.min()})")

        # Dijkstra from all source nodes (indices 0..n0-1)
        src_indices = np.arange(self.n0)
        dist_mat, pred_mat = dijkstra(
            graph, directed=False, indices=src_indices, return_predecessors=True
        )
        # dist_mat: (n0, N)  pred_mat: (n0, N)

        # Extract (n0, n1) cost matrix: source i → target j
        cost = dist_mat[:, self.n0 : self.n0 + self.n1].copy()

        inf_mask = np.isinf(cost)
        n_disconnected = int(inf_mask.sum())
        if n_disconnected > 0:
            frac = n_disconnected / (self.n0 * self.n1)
            print(f"    {n_disconnected}/{self.n0 * self.n1} source/target pairs disconnected ({frac:.1%}); using direct angular fallback")
            Y0_np = self.Y0.detach().cpu().numpy()
            Y1_np = self.Y1.detach().cpu().numpy()
            cos_direct = np.clip(Y0_np @ Y1_np.T, -1 + 1e-6, 1 - 1e-6)
            direct_ang = np.arccos(cos_direct)
            cost = np.where(inf_mask, direct_ang, cost)

        self.cost_matrix = cost.astype(np.float32)  # (n0, n1)
        self.predecessors = pred_mat  # (n0, N)
        self._disconnected_mask = inf_mask  # (n0, n1)
        self._graph_nnz = int(graph.nnz)

    def reconstruct_path(self, src_local, tgt_local):
        """
        src_local ∈ [0, n0), tgt_local ∈ [0, n1)
        Returns list of node indices into self.all_nodes (src first, tgt last).
        Disconnected pairs return a two-node direct path.
        """
        if self._disconnected_mask[src_local, tgt_local]:
            return [src_local, self.n0 + tgt_local]

        tgt_global = self.n0 + tgt_local
        path = [tgt_global]
        node = tgt_global
        max_hops = self.n0 + self.n1
        while node != src_local:
            parent = int(self.predecessors[src_local, node])
            if parent < 0 or max_hops <= 0:
                return [src_local, tgt_global]
            path.append(parent)
            node = parent
            max_hops -= 1
        return list(reversed(path))

    def sample(self, src_idx, tgt_idx, t):
        """
        src_idx, tgt_idx: (B,) numpy int arrays (local indices)
        t: (B, 1) torch tensor on device, in [0, 1]
        Returns: z_t (B, D), v_t (B, D) on device — polyline position and
        arc-length-parameterized tangent velocity.
        """
        device = t.device
        B = len(src_idx)

        # Reconstruct paths (Python loop is fine — O(path_len) per sample)
        paths = [self.reconstruct_path(int(s), int(g)) for s, g in zip(src_idx, tgt_idx)]
        path_lens = [len(p) for p in paths]
        max_len = max(path_lens)

        # Pad with last node so padded segments have zero length
        padded = np.empty((B, max_len), dtype=np.int64)
        for i, p in enumerate(paths):
            padded[i, : len(p)] = p
            padded[i, len(p) :] = p[-1]

        padded_t = torch.from_numpy(padded).to(device)  # (B, M)
        path_lens_t = torch.tensor(path_lens, dtype=torch.long, device=device)  # (B,)

        # Gather waypoint coordinates
        waypoints = self.all_nodes.to(device)[padded_t]  # (B, M, D)

        # Segment angular lengths between consecutive waypoints
        wa = waypoints[:, :-1]  # (B, M-1, D)
        wb = waypoints[:, 1:]  # (B, M-1, D)
        cos_seg = (wa * wb).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)  # (B, M-1)
        seg_lens = torch.acos(cos_seg)  # (B, M-1)

        # Mask padded segments (index k is valid iff k < path_len - 1)
        seg_idx = torch.arange(max_len - 1, device=device)  # (M-1,)
        valid = seg_idx.unsqueeze(0) < (path_lens_t - 1).unsqueeze(1)  # (B, M-1)
        seg_lens = seg_lens * valid.float()

        # Cumulative arc length with leading 0: (B, M)
        zeros = torch.zeros(B, 1, device=device)
        cum = torch.cat([zeros, seg_lens.cumsum(dim=1)], dim=1)

        # Total arc length at the end of each valid path
        total = cum.gather(1, (path_lens_t - 1).unsqueeze(1)).clamp(min=1e-8)  # (B, 1)

        # Target arc length = t * total
        target_arc = t * total  # (B, 1)

        # Find segment k s.t. cum[k] <= target_arc < cum[k+1]
        seg_k = torch.searchsorted(cum, target_arc, right=True) - 1  # (B, 1)
        max_seg = (path_lens_t - 2).clamp(min=0).unsqueeze(1)
        seg_k = torch.minimum(seg_k, max_seg).clamp(min=0)

        # Segment start / end cumulative lengths
        seg_start = cum.gather(1, seg_k)  # (B, 1)
        seg_end = cum.gather(1, seg_k + 1)  # (B, 1)
        seg_len = (seg_end - seg_start).clamp(min=1e-8)  # (B, 1)

        # Local parameter s∈[0,1] within the selected segment
        s = ((target_arc - seg_start) / seg_len).clamp(0.0, 1.0)  # (B, 1)

        # Gather the two waypoints defining the selected segment
        D = waypoints.shape[-1]
        idx_a = seg_k.unsqueeze(-1).expand(-1, -1, D)  # (B, 1, D)
        idx_b = (seg_k + 1).unsqueeze(-1).expand(-1, -1, D)
        wa_sel = waypoints.gather(1, idx_a).squeeze(1)  # (B, D)
        wb_sel = waypoints.gather(1, idx_b).squeeze(1)

        # Local SLERP within segment
        cos_omega = (wa_sel * wb_sel).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        z_t = (
            torch.sin((1 - s) * omega) / sin_omega * wa_sel
            + torch.sin(s * omega) / sin_omega * wb_sel
        )
        z_t = z_t / z_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Velocity wrt local s (d/ds SLERP), then chain-rule to global t:
        #   s = (t * total - seg_start) / seg_len  →  ds/dt = total / seg_len
        dz_ds = omega * (
            -torch.cos((1 - s) * omega) / sin_omega * wa_sel
            + torch.cos(s * omega) / sin_omega * wb_sel
        )
        v_t = dz_ds * (total / seg_len)

        # Project to tangent space at z_t (kills any numerical normal component)
        v_t = v_t - (v_t * z_t).sum(dim=-1, keepdim=True) * z_t

        return z_t, v_t


class PolylinePosSLERPVelInterpolant:
    """
    Hybrid: polyline position targets, SLERP velocity targets.

    Rationale (see polyline_diagnostic.html): the pure polyline interpolant
    creates a multi-valued regression target because OT pairs share interior
    hub cells with conflicting outgoing directions. Using the closed-form
    SLERP velocity (defined by the two endpoints, independent of the graph
    path) eliminates the conflict — at every shared cell the network sees a
    target that depends only on (src, tgt), not on which waypoint-level
    neighbor the graph chose next. Positions are still sampled on the data
    manifold via the polyline, testing the "training positions on the
    ribbon" hypothesis in isolation from the curved-velocity hypothesis.

    Caveat: the velocity no longer exactly integrates to the polyline path;
    the field behaves like SLERP globally but is evaluated on polyline
    positions. At t=0 it starts from y_src and at t=1 the target is y_tgt,
    so endpoints still match.
    """

    def __init__(self, poly_interp, slerp_interp):
        self.poly = poly_interp
        self.slerp = slerp_interp
        self.Y0 = slerp_interp.Y0
        self.Y1 = slerp_interp.Y1
        self.cost_matrix = poly_interp.cost_matrix
        self.device = slerp_interp.device

    def sample(self, src_idx, tgt_idx, t):
        z_poly, _ = self.poly.sample(src_idx, tgt_idx, t)
        _, v_slerp = self.slerp.sample(src_idx, tgt_idx, t)
        # Reproject SLERP velocity to the tangent space at the polyline position
        v = v_slerp - (v_slerp * z_poly).sum(dim=-1, keepdim=True) * z_poly
        return z_poly, v


class SparseAnchorGeodesic(PolylineGeodesic):
    """
    Sparse-anchor interpolant: piecewise SLERP through a handful of real
    training cells along the graph shortest path, instead of every waypoint.

    Motivation (see polyline_diagnostic.html): the full polyline fails because
    OT pairs share interior hubs, producing an 85% multi-valued-field conflict
    rate. Reducing to n_anchors interior waypoints dramatically cuts the
    conflict rate (two pairs must snap to exactly the same cell AND have
    matching outgoing direction) while still anchoring the interpolant to
    the data manifold at n_anchors points.

    The ablation is a bias/variance sweep: n_anchors=0 ≡ SLERP (no manifold
    info, fully consistent), n_anchors=full ≡ pure polyline (maximum manifold
    info, multi-valued supervision). The hypothesis is that a small interior
    count sits at the Pareto frontier.

    Anchor selection: n_anchors+2 equally spaced cumulative arc lengths along
    the full polyline, each snapped to the nearest real node. Endpoints are
    forced to the true src/tgt cells. If the path has fewer interior nodes
    than requested, returns the full polyline (no subsampling needed).

    Reuses a pre-built PolylineGeodesic so the kNN graph + Dijkstra are only
    computed once per transition.
    """

    def __init__(self, base_poly, n_anchors):
        # Bypass PolylineGeodesic.__init__ (would rebuild the graph) and
        # share all Dijkstra state with the base instance.
        self.device = base_poly.device
        self.Y0 = base_poly.Y0
        self.Y1 = base_poly.Y1
        self.n0 = base_poly.n0
        self.n1 = base_poly.n1
        self.all_nodes = base_poly.all_nodes
        self.predecessors = base_poly.predecessors
        self.cost_matrix = base_poly.cost_matrix  # graph is identical
        self._disconnected_mask = base_poly._disconnected_mask
        self._graph_nnz = base_poly._graph_nnz

        self.n_anchors = n_anchors
        self._all_nodes_np = base_poly.all_nodes.detach().cpu().numpy()

    def reconstruct_path(self, src_local, tgt_local):
        full = PolylineGeodesic.reconstruct_path(self, src_local, tgt_local)
        if self.n_anchors == 0:
            return [full[0], full[-1]]
        if len(full) <= 2 or (len(full) - 2) <= self.n_anchors:
            return full  # not enough interior waypoints to subsample

        pts = self._all_nodes_np[full]  # (L, D)
        cos_seg = np.clip((pts[:-1] * pts[1:]).sum(-1), -1 + 1e-6, 1 - 1e-6)
        seg_lens = np.arccos(cos_seg)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])  # (L,)
        total = cum[-1]
        if total < 1e-8:
            return [full[0], full[-1]]

        # Equally spaced arc-length targets, endpoints inclusive
        targets = np.linspace(0.0, total, self.n_anchors + 2)
        snap = np.array([int(np.argmin(np.abs(cum - t))) for t in targets])
        snap[0] = 0
        snap[-1] = len(full) - 1

        chosen = [full[snap[0]]]
        for i in range(1, len(snap)):
            node = full[snap[i]]
            if node != chosen[-1]:
                chosen.append(node)
        return chosen


# ─────────────────────────────────────────────────────────────────────────────
# 5. OT COUPLING
# ─────────────────────────────────────────────────────────────────────────────


def compute_sphere_cost_matrix(Y0, Y1):
    """Squared geodesic cost matrix between all pairs on the positive orthant.

    Fisher Flow (Davis et al., Prop. 2) uses c(x,y) = d²(x,y) for optimal
    transport — the W₂ cost — which gives the unique constant-speed geodesic
    interpolant. d(x,y) = arccos(⟨x,y⟩) on the unit sphere.
    Computed on whatever device the inputs live on; returned as numpy for POT.
    """
    cos_sim = (Y0 @ Y1.T).clamp(-1 + 1e-6, 1 - 1e-6)
    arc = torch.acos(cos_sim)
    return (arc ** 2).cpu().numpy()


def estimate_rbf_sigma(cells, k=50):
    """
    Bandwidth for an RBF kernel on the sphere: median arc distance to the
    k-th nearest neighbor among the training cells. Much more robust in
    high dimension than Silverman's rule.
    """
    from sklearn.neighbors import NearestNeighbors
    A = cells.detach().cpu().numpy()
    k_eff = min(k + 1, len(A))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
    nn.fit(A)
    dists, _ = nn.kneighbors(A)
    cos_sim = np.clip(1.0 - dists[:, 1:], -1 + 1e-6, 1 - 1e-6)
    arcs = np.arccos(cos_sim)  # (N, k)
    return float(np.median(arcs[:, -1]))


def rbf_spherical_score(z, cells, sigma):
    """
    Non-parametric Riemannian score of the data density on the sphere at
    query positions z. Uses an isotropic RBF kernel on arc-length distance:

        p(z) ∝ Σᵢ exp(-arc(z, cᵢ)² / (2σ²))
        score(z) = ∇ log p(z) = (1/σ²) · Σᵢ wᵢ(z) · log_z(cᵢ)

    where wᵢ are softmax-normalized kernel weights and log_z is the sphere
    log map. The score is zero on the ribbon (cells surround z symmetrically)
    and points toward the ribbon when z drifts off it — crucially, no pull
    toward the global mode because on-ribbon gradient flattens to zero.

    Factored implementation avoids an (B, N, D) intermediate tensor:
        score(z) = (1/σ²) · [ (w·a) @ cells  -  ((w·a·cos_ω).sum) · z ]

    where a = ω/sin(ω) is the exact log-map scale.

    Parameters
    ----------
    z: (B, D) unit vectors on the sphere (tangent basepoints)
    cells: (N, D) training cells on the sphere
    sigma: scalar bandwidth (arc-length units)

    Returns
    -------
    score: (B, D) tangent vectors at each z
    """
    cos_sim = (z @ cells.T).clamp(-1 + 1e-6, 1 - 1e-6)  # (B, N)
    arc = torch.acos(cos_sim)  # (B, N)

    # Softmax-normalized RBF weights (numerically stable)
    log_w = -(arc ** 2) / (2 * sigma * sigma)
    w = torch.softmax(log_w, dim=-1)  # (B, N)

    # Exact log-map scale a = ω / sin(ω); limit 1 at ω = 0
    sin_arc = torch.sin(arc)
    a = torch.where(arc < 1e-6, torch.ones_like(arc), arc / sin_arc.clamp(min=1e-8))

    wa = w * a  # (B, N)
    weighted_cells = wa @ cells  # (B, D)
    scalar_term = (wa * cos_sim).sum(dim=-1, keepdim=True)  # (B, 1)
    score = (weighted_cells - scalar_term * z) / (sigma * sigma)

    # Tangent projection (numerical cleanup)
    score = score - (score * z).sum(dim=-1, keepdim=True) * z
    return score


def ot_coupling(cost_matrix, n_samples, emd_max_pool=15000):
    """
    Solve OT problem and sample coupled pairs.
    Returns indices (src_idx, tgt_idx) of coupled pairs.

    Uses exact EMD when max(n, m) <= emd_max_pool (sharp permutation plan,
    best training signal), falls back to Sinkhorn above that threshold for
    tractability. Historically this always used Sinkhorn, which introduces
    entropic smoothing into the coupling — fine at high dim/N where EMD is
    intractable, but degrades training signal when EMD would have worked.
    """
    n, m = cost_matrix.shape
    if not HAS_POT:
        # Uniform random pairing fallback
        src = np.random.randint(0, n, n_samples)
        tgt = np.random.randint(0, m, n_samples)
        return src, tgt

    a = np.ones(n) / n
    b = np.ones(m) / m

    if max(n, m) <= emd_max_pool:
        # Exact EMD: sharp permutation plan
        T = ot.emd(a, b, cost_matrix)
    else:
        # Stabilized Sinkhorn at small ε: near-EMD sharpness, log-domain
        # stable, much better at high K than regular Sinkhorn with large ε.
        eps = 0.005 * cost_matrix.max()
        T = ot.sinkhorn(
            a, b, cost_matrix, reg=eps,
            method="sinkhorn_stabilized",
            numItermax=2000, stopThr=1e-7,
        )
    T = T / T.sum()  # normalize to joint distribution

    # Sample from the transport plan
    T_flat = T.flatten()
    T_flat = np.maximum(T_flat, 0)
    T_flat = T_flat / T_flat.sum()
    flat_idx = np.random.choice(len(T_flat), size=n_samples, p=T_flat)
    src_idx = flat_idx // m
    tgt_idx = flat_idx % m
    return src_idx, tgt_idx


# ─────────────────────────────────────────────────────────────────────────────
# 5. NEURAL NETWORK
# ─────────────────────────────────────────────────────────────────────────────


class FlowNet(nn.Module):
    """
    MLP backbone predicting a tangent velocity vector. Input: (z_t, t)
    concatenated. Output is projected onto the tangent space at z_t inside
    the training loop.
    """

    def __init__(self, D, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(D + 1, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)
        self.D = D

    def forward(self, zt, t):
        # t: scalar or (B,) -> (B, 1)
        if t.dim() == 0:
            t = t.expand(zt.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        x = torch.cat([zt, t], dim=-1)
        return self.net(x)


class SigmaEmbedding(nn.Module):
    """
    Random Fourier features for noise level conditioning, à la DDPM/EDM.

    Replaces raw log_sigma scalar input with a rich sinusoidal embedding.
    Frequencies are log-spaced over the range typical for log_sigma values
    in [-4, 0] (i.e., σ ∈ [0.018, 1.0]).

    Input: log_sigma, shape (B,) or (B, 1)
    Output: (B, emb_dim) embedding vector
    """

    def __init__(self, emb_dim=64, num_freqs=None):
        super().__init__()
        num_freqs = num_freqs or (emb_dim // 2)
        # Log-spaced frequencies, roughly [0.5, 50] rad/unit
        freqs = 2 * np.pi * torch.exp(
            torch.linspace(float(np.log(0.1)), float(np.log(50)), num_freqs)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * num_freqs, emb_dim)

    def forward(self, log_sigma):
        if log_sigma.dim() == 0:
            log_sigma = log_sigma.view(1)
        elif log_sigma.dim() == 2:
            log_sigma = log_sigma.squeeze(-1)
        # log_sigma: (B,)
        s = log_sigma.unsqueeze(-1) * self.freqs.view(1, -1)  # (B, num_freqs)
        feats = torch.cat([s.sin(), s.cos()], dim=-1)  # (B, 2*num_freqs)
        return self.proj(feats)  # (B, emb_dim)


class RiemannianScoreNet(nn.Module):
    """
    Parametric score estimator on the sphere. Input: (z, log_sigma). Output:
    tangent vector at z approximating ∇log p_σ(z), where p_σ is the noise-
    convolved density of the training cell cloud at scale σ.

    Improvements (vs the original Song & Ermon NCSN-style MLP):
      - SigmaEmbedding (Fourier features) instead of raw log_sigma scalar
      - Wider multi-scale range [σ=0.02, σ=1.0]
      - Compatible with lognormal σ sampling (EDM schedule)

    Tangent projection is applied inside forward() so the output is always
    a valid tangent vector at z.
    """

    def __init__(self, D, hidden=256, depth=4, sigma_emb_dim=64):
        super().__init__()
        self.sigma_emb = SigmaEmbedding(emb_dim=sigma_emb_dim)
        layers = [nn.Linear(D + sigma_emb_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)
        self.D = D

    def forward(self, z, log_sigma):
        if log_sigma.dim() == 0:
            log_sigma = log_sigma.expand(z.shape[0])
        emb = self.sigma_emb(log_sigma)  # (B, emb_dim)
        x = torch.cat([z, emb], dim=-1)
        raw = self.net(x)
        # Tangent projection at z
        return raw - (raw * z).sum(dim=-1, keepdim=True) * z


class TimeEmbedding(nn.Module):
    """
    Random Fourier features for trajectory time t ∈ [0, 1]. Mirror of
    SigmaEmbedding but with a lower frequency range appropriate for unit
    intervals: log-spaced from 0.5 to 16 cycles per unit. The lowest
    harmonic resolves the 0.25-wide developmental stage intervals.

    Input: t, shape (B,), (B, 1), or scalar. Output: (B, emb_dim).
    """

    def __init__(self, emb_dim=64, num_freqs=None):
        super().__init__()
        num_freqs = num_freqs or (emb_dim // 2)
        freqs = 2 * np.pi * torch.exp(
            torch.linspace(float(np.log(0.5)), float(np.log(16.0)), num_freqs)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * num_freqs, emb_dim)

    def forward(self, t):
        if t.dim() == 0:
            t = t.view(1)
        elif t.dim() == 2:
            t = t.squeeze(-1)
        # t: (B,)
        s = t.unsqueeze(-1) * self.freqs.view(1, -1)  # (B, num_freqs)
        feats = torch.cat([s.sin(), s.cos()], dim=-1)  # (B, 2*num_freqs)
        return self.proj(feats)  # (B, emb_dim)


class TimedRiemannianScoreNet(nn.Module):
    """
    Time-conditioned parametric score on the sphere.

    forward(z, log_sigma, t) -> tangent at z approximating
        ∇ log p_σ(z | developmental stage = t)

    Addresses the core limitation of RiemannianScoreNet: the unconditional
    score pulls toward the cell cloud centroid (marginal density gradient)
    regardless of which developmental stage the flow should be at. The
    time-conditioned variant learns stage-specific densities so the score
    points toward the correct target stage during multi-marginal flow
    training.

    Same tangent projection and MLP architecture as RiemannianScoreNet,
    but with an additional TimeEmbedding concatenated into the input.
    """

    def __init__(self, D, hidden=256, depth=4, sigma_emb_dim=64, time_emb_dim=64):
        super().__init__()
        self.sigma_emb = SigmaEmbedding(emb_dim=sigma_emb_dim)
        self.time_emb = TimeEmbedding(emb_dim=time_emb_dim)
        in_dim = D + sigma_emb_dim + time_emb_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)
        self.D = D

    def forward(self, z, log_sigma, t):
        if log_sigma.dim() == 0:
            log_sigma = log_sigma.expand(z.shape[0])
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        elif t.dim() == 2:
            t = t.squeeze(-1)
        emb_s = self.sigma_emb(log_sigma)  # (B, sigma_emb_dim)
        emb_t = self.time_emb(t)  # (B, time_emb_dim)
        x = torch.cat([z, emb_s, emb_t], dim=-1)
        raw = self.net(x)
        return raw - (raw * z).sum(dim=-1, keepdim=True) * z


def sample_log_sigma_lognormal(batch_size, device, sigma_min, sigma_max,
                                log_mean=-1.2, log_std=1.2):
    """
    EDM-style lognormal σ sampling, clipped to [sigma_min, sigma_max].
    Concentrates training samples near σ ≈ exp(log_mean) where DSM is
    most informative, rather than log-uniform across the whole range.

    Default (mean=-1.2, std=1.2) peaks at σ ≈ 0.3.

    Returns log_sigma of shape (batch_size, 1).
    """
    log_sigma = torch.randn(batch_size, 1, device=device) * log_std + log_mean
    return log_sigma.clamp(float(np.log(sigma_min)), float(np.log(sigma_max)))


def sphere_brownian_perturb(c, sigma, n_steps=3):
    """
    Multi-step Euler-Maruyama approximation to Brownian motion on the unit
    sphere, starting at c with total noise scale σ. More faithful to the
    heat kernel (de Bortoli et al. 2022) than a single Gaussian-tangent
    retraction, especially at larger σ where the single-step approximation
    biases samples toward the tangent plane.

    Each of n_steps applies a Gaussian tangent perturbation with variance
    σ²/n_steps and retracts via Exp. The total tangent-space variance
    accumulates to σ² (approximately; manifold curvature introduces a
    small correction for large σ).

    c: (B, D) unit vectors
    sigma: (B, 1) per-sample noise scale
    Returns z: (B, D) perturbed points on the sphere.
    """
    z = c
    # sqrt(n_steps) so that K steps of variance σ²/K accumulate to σ²
    step_sigma = sigma / (n_steps ** 0.5)
    for _ in range(n_steps):
        eps = torch.randn_like(z)
        eps = eps - (eps * z).sum(dim=-1, keepdim=True) * z  # tangent at z
        v = step_sigma * eps
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z = torch.cos(v_norm) * z + torch.sin(v_norm) * v / v_norm
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return z


def product_sphere_brownian_perturb(c3, sigma, n_steps=3):
    """
    Per-component multi-step Brownian motion on a product of spheres
    (S^{K-1})^k, where c3 has shape (B, k, K) and sigma has shape (B, 1, 1).

    Returns z3 of shape (B, k, K).
    """
    z = c3
    step_sigma = sigma / (n_steps ** 0.5)
    for _ in range(n_steps):
        eps = torch.randn_like(z)
        eps = eps - (eps * z).sum(dim=-1, keepdim=True) * z
        v = step_sigma * eps
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        z = torch.cos(v_norm) * z + torch.sin(v_norm) * v / v_norm
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return z


def train_riemannian_score(
    cells,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    sigma_min=0.02,
    sigma_max=1.0,
    lognormal_mean=-1.2,
    lognormal_std=1.2,
    n_brownian_steps=3,
    hidden=256,
    depth=4,
    label="ScoreNet",
):
    """
    Denoising score matching on the sphere (Song & Ermon, de Bortoli et al.).

    Improvements vs naive DSM:
      1. Wider σ range [sigma_min=0.02, sigma_max=1.0] — network sees
         multi-scale behavior instead of a narrow band.
      2. Lognormal σ sampling (EDM, Karras et al. 2022) with
         log_sigma ~ N(log_mean, log_std²) clipped to [σ_min, σ_max].
         Concentrates samples near σ ≈ 0.3 where DSM is most informative.
      3. Multi-step Brownian perturbation (n_brownian_steps Euler steps)
         instead of a single Gaussian-tangent retraction. More faithful to
         the heat kernel on the sphere at large σ.
      4. Sinusoidal (Fourier) embedding of log_sigma via SigmaEmbedding,
         inside RiemannianScoreNet. Richer conditioning than raw scalar.

    Target: log_z(c) / σ² (Riemannian DSM). Loss: σ²-weighted MSE so the
    training signal is balanced across noise scales.
    """
    model = RiemannianScoreNet(D, hidden=hidden, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = len(cells)
    cells_dev = cells.to(DEVICE)

    losses = []
    for i in range(n_iters):
        idx = np.random.randint(0, N, size=batch_size)
        c = cells_dev[idx]  # (B, D)

        # Lognormal σ sampling (clipped to valid range)
        log_sigma = sample_log_sigma_lognormal(
            batch_size, DEVICE, sigma_min, sigma_max,
            log_mean=lognormal_mean, log_std=lognormal_std,
        )
        sigma = log_sigma.exp()  # (B, 1)

        # Multi-step Brownian perturbation (approx heat kernel)
        z = sphere_brownian_perturb(c, sigma, n_steps=n_brownian_steps)

        # Target: log_z(c) / σ²
        cos_omega = (z * c).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        log_z_to_c = (omega / sin_omega) * (c - cos_omega * z)
        log_z_to_c = log_z_to_c - (log_z_to_c * z).sum(dim=-1, keepdim=True) * z
        target = log_z_to_c / (sigma * sigma)

        s_pred = model(z, log_sigma.squeeze(1))

        # σ²-weighted MSE: cancels the 1/σ² target scale
        loss = (((s_pred - target) * sigma) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {i:4d}  loss={loss.item():.4f}")

    model.eval()
    return model, losses


def train_timed_riemannian_score(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    sigma_min=0.02,
    sigma_max=1.0,
    lognormal_mean=-1.2,
    lognormal_std=1.2,
    n_brownian_steps=3,
    hidden=256,
    depth=4,
    label="TimedScoreNet",
):
    """
    Time-conditioned denoising score matching on the sphere.

    At each step:
      1. Pick a stage interval i ∈ [0, S-2] uniformly.
      2. Sample s_local ~ U(0, 1) per batch element.
      3. t = stage_times[i] + s_local * (stage_times[i+1] - stage_times[i]).
      4. Draw each batch element's clean cell c from stage i with prob
         (1 - s_local), else from stage i+1. This gives a valid mixture
         between the two marginals that the flow must pass through.
      5. Sample σ via lognormal, perturb c via multi-step sphere Brownian.
      6. Target: log_z(c) / σ², same as train_riemannian_score.
      7. Predict via TimedRiemannianScoreNet(z, log_sigma, t), σ²-weighted MSE.

    At inference (multi-marginal flow training), the timed score net is
    queried at (z_t, log_sigma_tensor, t_global) so it pulls z_t toward
    cells at the correct developmental stage, not the pooled centroid.

    Returns (model, losses) with model in eval mode.
    """
    S = len(stage_cells)
    assert S >= 2, "need at least 2 stages for a multi-marginal time-conditioned score"
    stage_cells_dev = [c.to(DEVICE) for c in stage_cells]

    model = TimedRiemannianScoreNet(D, hidden=hidden, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for it in range(n_iters):
        # Pick an interval and sample s_local ~ U(0, 1)
        i = np.random.randint(0, S - 1)
        s_local = torch.rand(batch_size, 1, device=DEVICE)  # (B, 1)
        t_start = stage_times[i]
        t_end = stage_times[i + 1]
        t_batch = t_start + s_local * (t_end - t_start)  # (B, 1)

        # Per-element coin flip: stage i with prob (1 - s_local), else stage i+1
        use_next = (torch.rand(batch_size, device=DEVICE) < s_local.squeeze(1))
        n_next = int(use_next.sum().item())
        n_this = batch_size - n_next

        cells_this = stage_cells_dev[i]
        cells_next = stage_cells_dev[i + 1]
        idx_this = torch.randint(0, len(cells_this), (n_this,), device=DEVICE)
        idx_next = torch.randint(0, len(cells_next), (n_next,), device=DEVICE)

        c = torch.empty(batch_size, D, device=DEVICE, dtype=cells_this.dtype)
        c[~use_next] = cells_this[idx_this]
        c[use_next] = cells_next[idx_next]

        # Lognormal σ
        log_sigma = sample_log_sigma_lognormal(
            batch_size, DEVICE, sigma_min, sigma_max,
            log_mean=lognormal_mean, log_std=lognormal_std,
        )
        sigma = log_sigma.exp()  # (B, 1)

        # Multi-step Brownian perturbation
        z = sphere_brownian_perturb(c, sigma, n_steps=n_brownian_steps)

        # DSM target: log_z(c) / σ²
        cos_omega = (z * c).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        log_z_to_c = (omega / sin_omega) * (c - cos_omega * z)
        log_z_to_c = log_z_to_c - (log_z_to_c * z).sum(dim=-1, keepdim=True) * z
        target = log_z_to_c / (sigma * sigma)

        s_pred = model(z, log_sigma.squeeze(1), t_batch.squeeze(1))

        loss = (((s_pred - target) * sigma) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {it:4d}  loss={loss.item():.4f}")

    model.eval()
    return model, losses


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────────────────────


def train_fisher_flow(
    interpolant,
    cost_matrix,
    D,
    n_iters=2000,
    batch_size=256,
    lr=3e-4,
    label="Fisher Flow",
    score_cells=None,
    sigma=None,
    alpha=0.0,
    score_net=None,
    score_net_sigma=0.1,
):
    """
    Riemannian flow matching on the positive orthant. Trains a tangent
    velocity network against the target velocity of a supplied interpolant
    (either SLERPInterpolant or PolylineGeodesic). The cost_matrix controls
    OT coupling and can differ from the interpolant's own geometry — that
    mismatch is the off-diagonal ablation.

    Score regularization: when alpha>0 and a score source is supplied, the
    velocity target is augmented at every z_t with

        v_target ← v_target + alpha * s(z_t)

    Two score sources are accepted (score_net takes precedence if both are
    given):

      * score_net: a trained RiemannianScoreNet. Evaluated at
        (z_t, score_net_sigma). Robust in high dim.
      * score_cells + sigma: non-parametric RBF-KDE on the training cell
        cloud. Closed-form but suffers curse of dimensionality in 1591-d.
    """
    model = FlowNet(D).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-sample OT pairs once to avoid resolving the transport plan each step
    n_pool = min(50000, cost_matrix.shape[0] * cost_matrix.shape[1])
    print(f"  Computing OT coupling for {label}...")
    ot_src, ot_tgt = ot_coupling(cost_matrix, n_pool)

    use_net = alpha > 0.0 and score_net is not None
    use_kde = alpha > 0.0 and score_cells is not None and sigma is not None and not use_net
    if use_net:
        score_net.eval()
        log_sigma_tensor = torch.full((batch_size,), float(np.log(score_net_sigma)), device=DEVICE)
        print(f"  Score regularization: alpha={alpha}, score_net sigma={score_net_sigma:.4f} (learned)")
    elif use_kde:
        print(f"  Score regularization: alpha={alpha}, sigma={sigma:.4f}, cells={len(score_cells)} (KDE)")

    losses = []
    for i in range(n_iters):
        idx = np.random.choice(len(ot_src), size=batch_size, replace=True)
        src_idx = ot_src[idx]
        tgt_idx = ot_tgt[idx]

        t = torch.rand(batch_size, 1, device=DEVICE)
        z_t, v_target = interpolant.sample(src_idx, tgt_idx, t)

        if use_net:
            with torch.no_grad():
                score = score_net(z_t, log_sigma_tensor)
            v_target = v_target + alpha * score
        elif use_kde:
            with torch.no_grad():
                score = rbf_spherical_score(z_t, score_cells, sigma)
            v_target = v_target + alpha * score

        v_pred_raw = model(z_t.detach(), t.squeeze(1))
        v_pred = v_pred_raw - (v_pred_raw * z_t).sum(dim=-1, keepdim=True) * z_t

        loss = ((v_pred - v_target.detach()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {i:4d}  loss={loss.item():.4f}")

    return model, losses


def compute_phate_cost_matrix(Y0, Y1, n_components=10, **kwargs):
    """
    OT cost matrix using squared PHATE potential distance.

    Delegates to the package implementation so the legacy CLI and Hydra CLI
    use the same pre-MDS PHATE potential cost.
    """
    from surf.ot.costs import compute_phate_cost_matrix as _compute_phate_cost_matrix

    return _compute_phate_cost_matrix(Y0, Y1, n_components=n_components, **kwargs)


def make_phate_cost_fn(**kwargs):
    """Legacy CLI shim for the package PHATE cost factory."""
    from surf.ot.costs import make_phate_cost_fn as _make_phate_cost_fn

    return _make_phate_cost_fn(**kwargs)


def compute_biharmonic_cost_matrix(Y0, Y1, knn=15, n_eig=50, weight_power=0.5):
    """
    OT cost matrix using biharmonic distance on a kNN graph built from the
    union of Y0 and Y1. Biharmonic distance = sum of (u_k(i) - u_k(j))^2 / λ_k^(2*weight_power)
    where (u_k, λ_k) are eigenvectors/eigenvalues of the graph Laplacian.

    weight_power controls how aggressively low-frequency eigenvectors are
    amplified:
      - weight_power=1 → true biharmonic (1/λ²), can collapse to 1-2 eigs
      - weight_power=0.5 → 1/λ weighting (diffusion distance-like)
      - weight_power=0 → uniform weighting over the top-k eigenvectors
    Default 0.5 gives a stable balance between global structure and noise.

    This is a smoothed, global version of graph shortest-path distance —
    more robust than raw Dijkstra and captures manifold structure via the
    spectrum of the Laplacian.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    Y0_np = Y0.detach().cpu().numpy()
    Y1_np = Y1.detach().cpu().numpy()
    n0, n1 = len(Y0_np), len(Y1_np)
    N = n0 + n1
    combined = np.vstack([Y0_np, Y1_np])

    # kNN graph with Gaussian weights on arc-length distance
    nn = NearestNeighbors(n_neighbors=knn + 1, metric="cosine")
    nn.fit(combined)
    dists, inds = nn.kneighbors(combined)
    cos_sim = np.clip(1.0 - dists, -1 + 1e-6, 1 - 1e-6)
    arc = np.arccos(cos_sim)  # (N, knn+1)
    sigma = np.median(arc[:, 1:])
    weights = np.exp(-(arc ** 2) / (2 * sigma ** 2))

    rows = np.repeat(np.arange(N), knn + 1)
    cols = inds.reshape(-1)
    data = weights.reshape(-1)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W.maximum(W.T)  # symmetrize
    # Normalized Laplacian
    d = np.asarray(W.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = diags(d_inv_sqrt)
    L = diags(np.ones(N)) - D_inv_sqrt @ W @ D_inv_sqrt

    # Smallest n_eig+1 eigenpairs (skip the first trivial 0 eigenvalue)
    k_eig = min(n_eig + 1, N - 2)
    eigvals, eigvecs = eigsh(L, k=k_eig, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order][1:]  # skip trivial
    eigvecs = eigvecs[:, order][:, 1:]

    # Spectral embedding with configurable weighting
    inv_lam = 1.0 / np.maximum(eigvals, 1e-8) ** weight_power
    biharm_emb = eigvecs * inv_lam[None, :]  # (N, n_eig)

    e0 = biharm_emb[:n0]
    e1 = biharm_emb[n0:]
    diff = e0[:, None, :] - e1[None, :, :]
    cost = (diff ** 2).sum(axis=-1).astype(np.float32)
    return cost


def compute_global_biharmonic_embedding(stage_cells, knn=15, n_eig=50, weight_power=0.5):
    """Compute a single biharmonic spectral embedding over ALL stages.

    Builds one kNN graph on the union of all stage cells, computes the
    normalized Laplacian, and returns the weighted spectral embedding.
    All per-hop OT couplings then use distances in this shared coordinate
    system, guaranteeing genuine global coherence across intervals.

    Returns (embeddings_per_stage, stage_sizes) where embeddings_per_stage[i]
    is the (N_i, n_eig) embedding for stage i.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh

    all_np = [Y.detach().cpu().numpy() for Y in stage_cells]
    sizes = [len(Y) for Y in all_np]
    combined = np.vstack(all_np)
    N = len(combined)

    nn = NearestNeighbors(n_neighbors=knn + 1, metric="cosine")
    nn.fit(combined)
    dists, inds = nn.kneighbors(combined)
    cos_sim = np.clip(1.0 - dists, -1 + 1e-6, 1 - 1e-6)
    arc = np.arccos(cos_sim)
    sigma = np.median(arc[:, 1:])
    weights = np.exp(-(arc ** 2) / (2 * sigma ** 2))

    rows = np.repeat(np.arange(N), knn + 1)
    cols = inds.reshape(-1)
    data = weights.reshape(-1)
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    W = W.maximum(W.T)
    d = np.asarray(W.sum(axis=1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = diags(d_inv_sqrt)
    L = diags(np.ones(N)) - D_inv_sqrt @ W @ D_inv_sqrt

    k_eig = min(n_eig + 1, N - 2)
    eigvals, eigvecs = eigsh(L, k=k_eig, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order][1:]
    eigvecs = eigvecs[:, order][:, 1:]

    inv_lam = 1.0 / np.maximum(eigvals, 1e-8) ** weight_power
    biharm_emb = eigvecs * inv_lam[None, :]

    embeddings = []
    offset = 0
    for sz in sizes:
        embeddings.append(biharm_emb[offset:offset + sz])
        offset += sz
    return embeddings


def make_global_biharmonic_cost_fn(stage_cells):
    """Return a cost_fn(Y0, Y1) that uses a precomputed global embedding.

    The returned function matches the signature of compute_biharmonic_cost_matrix
    but looks up rows from the precomputed global spectral embedding instead
    of building a per-pair graph. The stage_cells list is used at construction
    time to build the global embedding; the returned function identifies Y0/Y1
    by matching against stored stage references.
    """
    print("  Computing global biharmonic embedding across all stages...")
    embeddings = compute_global_biharmonic_embedding(stage_cells)
    stage_refs = [id(Y) for Y in stage_cells]
    emb_by_id = dict(zip(stage_refs, embeddings))

    def global_biharmonic_cost(Y0, Y1):
        e0 = emb_by_id.get(id(Y0))
        e1 = emb_by_id.get(id(Y1))
        if e0 is None or e1 is None:
            raise RuntimeError("Global biharmonic: Y0 or Y1 not found in precomputed stages. "
                               "Ensure stage_cells references are not modified after construction.")
        diff = e0[:, None, :] - e1[None, :, :]
        return (diff ** 2).sum(axis=-1).astype(np.float32)

    global_biharmonic_cost.__name__ = "global_biharmonic_cost"
    return global_biharmonic_cost


def train_forward_score_nets(
    stage_cells,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    sigma_min=0.05,
    sigma_max=0.7,
):
    """
    Train one forward-directed score net per stage interval.

    For interval i (between stages i and i+1), the score net is trained on
    cells from stages i+1, i+2, ..., S-1 only — the "future" stages. This
    ensures the score always pulls the trajectory forward along the
    developmental direction, never backward toward already-visited stages.

    Returns a list of S-1 trained RiemannianScoreNets (one per interval).
    """
    S = len(stage_cells)
    nets = []
    for i in range(S - 1):
        future_cells = torch.cat(stage_cells[i + 1:], dim=0)
        print(f"\n  Interval {i} (t∈[{i/(S-1):.2f}, {(i+1)/(S-1):.2f}]): "
              f"training on {len(future_cells)} cells from stages {i+1}..{S-1}")
        net, _ = train_riemannian_score(
            future_cells, D, n_iters=n_iters, batch_size=batch_size,
            lr=lr, sigma_min=sigma_min, sigma_max=sigma_max,
            label=f"ScoreNet[interval{i}]",
        )
        nets.append(net)
    return nets


def train_multi_marginal_flow(
    stage_cells,
    stage_times,
    D,
    n_iters=3000,
    batch_size=256,
    lr=3e-4,
    label="MultiMarginal",
    ot_subsample=2000,
    score_net=None,
    score_nets_per_interval=None,
    alpha=0.0,
    score_net_sigma=0.1,
    cost_fn=None,
    si_sigma=0.0,
    si_brownian_steps=3,
    biharm_beta=0.0,
    biharm_waypoints=False,
    global_biharm_embeddings=None,
):
    """
    Multi-marginal Fisher Flow Matching. Trains a single FlowNet that passes
    through S developmental stages as marginals at times stage_times[0..S-1].

    At each training step:
      1. Pick a random adjacent pair of stages (i, i+1).
      2. Sample t ~ Uniform(stage_times[i], stage_times[i+1]).
      3. Rescale to local [0, 1] within the interval and SLERP between
         OT-coupled pairs from stages i and i+1.
      4. Compute v_target in the global time parameterization (SLERP velocity
         scaled by 1/(t_{i+1} - t_i)), optionally augmented by alpha*score.

    This generalizes the pairwise Fisher Flow to a single velocity field
    continuous across all stages. Unlike LOO, every stage is supervised
    directly, so we test the method's ability to match held-out *cells*
    within each stage rather than reconstruct an unseen intermediate stage.

    Parameters
    ----------
    stage_cells: list of (N_i, D) torch tensors on DEVICE, one per stage,
        already mapped to the positive orthant.
    stage_times: list of floats in [0,1], one per stage, strictly increasing.
    score_net: optional single RiemannianScoreNet (global, all stages).
    score_nets_per_interval: optional list of S-1 RiemannianScoreNets, one
        per interval. When provided, interval i uses score_nets_per_interval[i]
        instead of score_net. This enables forward-directed score regularization
        where each interval's score only sees future stages.
    alpha: strength of the score regularizer.

    Returns
    -------
    (model, losses)
    """
    assert len(stage_cells) == len(stage_times)
    S = len(stage_cells)
    assert S >= 2
    for i in range(S - 1):
        assert stage_times[i] < stage_times[i + 1]

    model = FlowNet(D).to(DEVICE)
    if USE_AMP:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"  torch.compile failed ({e}); running uncompiled")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # EMA of model weights for stability (0.999 decay)
    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    warmup_iters = min(500, n_iters // 10)

    # Precompute OT couplings for each adjacent pair, once (stable over training).
    # We subsample each stage to ot_subsample cells for OT tractability.
    print(f"  Computing OT couplings for {S - 1} adjacent pairs...")
    stage_cells_sub = []
    rng = np.random.default_rng(0)
    for i, cells in enumerate(stage_cells):
        if len(cells) > ot_subsample:
            idx = rng.choice(len(cells), size=ot_subsample, replace=False)
            stage_cells_sub.append(cells[idx])
        else:
            stage_cells_sub.append(cells)
        print(f"    stage {i}: {len(stage_cells_sub[i])} cells (t={stage_times[i]:.2f})")

    cost_label = "sphere W₂"
    if cost_fn is not None:
        cost_label = cost_fn.__name__ if hasattr(cost_fn, "__name__") else "custom"
    print(f"  OT cost: {cost_label}")
    adj_couplings = []  # list of (ot_src, ot_tgt) numpy arrays
    for i in range(S - 1):
        Y0, Y1 = stage_cells_sub[i], stage_cells_sub[i + 1]
        if cost_fn is not None:
            cost = cost_fn(Y0, Y1)
        else:
            cost = compute_sphere_cost_matrix(Y0, Y1)
        n_pool = min(20000, len(Y0) * len(Y1))
        os_, ot_ = ot_coupling(cost, n_pool)
        adj_couplings.append((os_, ot_))

    use_global_score = alpha > 0.0 and score_net is not None and score_nets_per_interval is None
    use_interval_score = alpha > 0.0 and score_nets_per_interval is not None
    use_score = use_global_score or use_interval_score

    log_sigma_tensor = torch.full((batch_size,), float(np.log(score_net_sigma)), device=DEVICE) if use_score else None

    if use_global_score:
        score_net.eval()
        print(f"  Score regularization: alpha={alpha}, sigma={score_net_sigma:.4f} (global)")
    elif use_interval_score:
        for sn in score_nets_per_interval:
            if isinstance(sn, nn.Module):
                sn.eval()
        print(f"  Score regularization: alpha={alpha}, sigma={score_net_sigma:.4f} (per-interval forward-directed)")

    # Interpret si_sigma as the target Brownian arc length (radians on the
    # sphere). sphere_brownian_perturb's `sigma` is a per-component scale
    # applied to an ambient Gaussian whose tangent projection has norm
    # ~sqrt(D-1), so the effective arc length is per_component_sigma *
    # sqrt(D-1). Invert this to get the per-component scale that produces
    # the requested arc. This makes si_sigma dimensionally interpretable
    # as "typical arc distance the noise should cover", independent of D.
    si_per_component_sigma = 0.0
    if si_sigma > 0.0:
        si_per_component_sigma = float(si_sigma) / float(np.sqrt(max(D - 1, 1)))
        print(f"  Stochastic interpolant: si_sigma={si_sigma:.4f} rad (target arc), "
              f"per_component={si_per_component_sigma:.6f}, n_brownian_steps={si_brownian_steps}")

    # Biharmonic velocity blending (Approach A): blend SLERP velocity with
    # the biharmonic spectral-gradient direction from Chen & Lipman (2024).
    # The spectral embedding maps each training cell to R^k; for a SLERP
    # midpoint x_t we extend the embedding via kNN interpolation, then
    # compute the spectral direction toward the target x_1.
    use_biharm_vel = biharm_beta > 0.0 and global_biharm_embeddings is not None
    # Biharmonic waypoints (Approach B): piecewise SLERP through a
    # biharmonic-midpoint cell, doubling the number of SLERP segments.
    use_biharm_wp = biharm_waypoints and global_biharm_embeddings is not None

    biharm_emb_tensors = None
    biharm_knn_idx = None
    biharm_knn_weights = None
    waypoint_idx_per_interval = None

    if use_biharm_vel or use_biharm_wp:
        biharm_emb_tensors = [torch.from_numpy(e.astype(np.float32)).to(DEVICE)
                              for e in global_biharm_embeddings]
        # Build kNN index for out-of-sample spectral embedding extension
        # (needed for Approach A to compute Φ(x_t) at SLERP midpoints)
        if use_biharm_vel:
            all_sub_cells = torch.cat(stage_cells_sub, dim=0)
            all_sub_emb = torch.cat(biharm_emb_tensors, dim=0)
            print(f"  Biharmonic velocity blend: beta={biharm_beta:.3f}")

        if use_biharm_wp:
            # For each adjacent pair, precompute the biharmonic midpoint cell
            # for every OT-coupled pair. The midpoint is the training cell
            # whose spectral embedding is closest to the average of the
            # source and target embeddings.
            all_train = torch.cat(stage_cells_sub, dim=0)
            all_emb = torch.cat(biharm_emb_tensors, dim=0)
            waypoint_idx_per_interval = []
            for iv in range(S - 1):
                os_, ot_ = adj_couplings[iv]
                e0 = biharm_emb_tensors[iv][os_]  # (n_pairs, k)
                e1 = biharm_emb_tensors[iv + 1][ot_]
                mid_emb = (e0 + e1) / 2  # (n_pairs, k)
                # Find closest training cell in spectral space
                dists = torch.cdist(mid_emb, all_emb)  # (n_pairs, N_all)
                wp_idx = dists.argmin(dim=1)  # (n_pairs,)
                waypoint_idx_per_interval.append(wp_idx)
            print(f"  Biharmonic waypoints: piecewise SLERP through spectral midpoints")

    losses = []
    for it in range(n_iters):
        # Pick a random adjacent pair for each batch element. Simplest: uniform
        # over intervals, then fill the batch from that single interval. This
        # keeps the SLERP math batched.
        i = np.random.randint(0, S - 1)
        Y0 = stage_cells_sub[i]
        Y1 = stage_cells_sub[i + 1]
        ot_src_i, ot_tgt_i = adj_couplings[i]
        t_start = stage_times[i]
        t_end = stage_times[i + 1]
        dt_interval = t_end - t_start

        idx = np.random.choice(len(ot_src_i), size=batch_size, replace=True)
        s_idx = ot_src_i[idx]
        g_idx = ot_tgt_i[idx]

        # Local SLERP parameter in [0, 1] and global t
        s_local = torch.rand(batch_size, 1, device=DEVICE)
        t_global = t_start + dt_interval * s_local  # (B, 1) in [t_start, t_end]

        y0 = Y0[s_idx]
        y1 = Y1[g_idx]

        # Approach B: piecewise SLERP through biharmonic midpoint waypoint
        if use_biharm_wp:
            all_train_cells = torch.cat(stage_cells_sub, dim=0)
            wp_global_idx = waypoint_idx_per_interval[i][idx]
            y_mid = all_train_cells[wp_global_idx]
            # Two segments: [y0→y_mid] for s_local<0.5, [y_mid→y1] for s_local≥0.5
            first_half = (s_local < 0.5).squeeze(-1)
            # Remap s_local to sub-segment local parameter
            s_sub = torch.where(first_half.unsqueeze(-1), 2 * s_local, 2 * s_local - 1)
            seg_start = torch.where(first_half.unsqueeze(-1), y0, y_mid)
            seg_end = torch.where(first_half.unsqueeze(-1), y_mid, y1)
            # Effective interval dt: each sub-segment covers half the interval
            dt_sub = dt_interval / 2.0
        else:
            s_sub = s_local
            seg_start = y0
            seg_end = y1
            dt_sub = dt_interval

        # SLERP position + local velocity (magnitude = omega in s_local time)
        cos_omega = (seg_start * seg_end).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        w0 = torch.sin((1 - s_sub) * omega) / sin_omega
        w1 = torch.sin(s_sub * omega) / sin_omega
        z_t = w0 * seg_start + w1 * seg_end
        z_t = z_t / z_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # Stochastic interpolant: perturb the SLERP position with Riemannian
        # Brownian noise so the velocity field is supervised on a neighborhood
        # of the clean SLERP arc, not just the arc itself. First-order
        # parallel transport of v_target is handled by the existing
        # `v_local - (v_local * z_t) * z_t` tangent projection below, which
        # now runs at the noisy z_t.
        if si_sigma > 0.0:
            with torch.no_grad():
                z_t_clean = z_t
                z_t = sphere_brownian_perturb(z_t, si_per_component_sigma, n_steps=si_brownian_steps)
                if it == 0:
                    cos_disp = (z_t * z_t_clean).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
                    arc_displacement = torch.acos(cos_disp)
                    ref_cells = Y0[:256]
                    arc_to_nearest_clean = torch.acos(
                        (z_t_clean @ ref_cells.T).clamp(-1 + 1e-6, 1 - 1e-6)
                    ).min(dim=-1).values
                    arc_to_nearest_noisy = torch.acos(
                        (z_t @ ref_cells.T).clamp(-1 + 1e-6, 1 - 1e-6)
                    ).min(dim=-1).values
                    print(f"    [diag@iter0, SI] arc_displacement(clean→noisy)  "
                          f"mean={arc_displacement.mean():.4f}  "
                          f"p95={arc_displacement.quantile(0.95):.4f}")
                    print(f"    [diag@iter0, SI] arc-to-nearest(clean)  "
                          f"mean={arc_to_nearest_clean.mean():.4f}  "
                          f"p95={arc_to_nearest_clean.quantile(0.95):.4f}")
                    print(f"    [diag@iter0, SI] arc-to-nearest(noisy)  "
                          f"mean={arc_to_nearest_noisy.mean():.4f}  "
                          f"p95={arc_to_nearest_noisy.quantile(0.95):.4f}")

        v_local = omega * (
            -torch.cos((1 - s_sub) * omega) / sin_omega * seg_start
            + torch.cos(s_sub * omega) / sin_omega * seg_end
        )
        v_local = v_local - (v_local * z_t).sum(dim=-1, keepdim=True) * z_t
        v_target = v_local / dt_sub

        # Approach A: blend SLERP velocity with biharmonic spectral direction.
        # The biharmonic direction at z_t toward y1 is computed as the
        # sphere log-map from z_t to y1, scaled to match the SLERP velocity
        # magnitude. This teaches the velocity network to follow the data
        # manifold geometry (via the spectral embedding) rather than the
        # great-circle arc.
        if use_biharm_vel:
            with torch.no_grad():
                # log-map from z_t to y1: the geodesic tangent pointing at y1
                cos_zy = (z_t * y1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
                theta_zy = torch.acos(cos_zy)
                sin_zy = torch.sin(theta_zy).clamp(min=1e-8)
                log_zt_y1 = (theta_zy / sin_zy) * (y1 - cos_zy * z_t)
                # Scale to match SLERP velocity magnitude for stable blending
                v_slerp_norm = v_target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                log_norm = log_zt_y1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                v_biharm_dir = log_zt_y1 * (v_slerp_norm / log_norm)
                v_target = (1 - biharm_beta) * v_target + biharm_beta * v_biharm_dir

        if use_score:
            with torch.no_grad():
                cur_score_net = score_nets_per_interval[i] if use_interval_score else score_net
                if isinstance(cur_score_net, TimedRiemannianScoreNet):
                    score = cur_score_net(z_t, log_sigma_tensor, t_global.squeeze(1))
                else:
                    score = cur_score_net(z_t, log_sigma_tensor)

                if it == 0:
                    v_slerp_norm = v_target.norm(dim=-1)
                    score_norm = (alpha * score).norm(dim=-1)
                    v_slerp_hat = v_target / v_slerp_norm.unsqueeze(-1).clamp(min=1e-8)
                    score_hat = score / score.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    cos_align = (v_slerp_hat * score_hat).sum(dim=-1)
                    any_cells = Y0[:256]
                    cos_to_cells = (z_t @ any_cells.T).clamp(-1 + 1e-6, 1 - 1e-6)
                    arc_to_nearest = torch.acos(cos_to_cells).min(dim=-1).values
                    mode = "forward-interval" if use_interval_score else "global"
                    print(f"    [diag@iter0, {mode}] ||v_slerp||  mean={v_slerp_norm.mean():.3f}  "
                          f"p95={v_slerp_norm.quantile(0.95):.3f}")
                    print(f"    [diag@iter0, {mode}] ||α·score|| mean={score_norm.mean():.3f}  "
                          f"p95={score_norm.quantile(0.95):.3f}  "
                          f"ratio={score_norm.mean() / v_slerp_norm.mean():.3f}")
                    print(f"    [diag@iter0, {mode}] cos(v_slerp, score)  "
                          f"mean={cos_align.mean():.3f}  std={cos_align.std():.3f}")
                    print(f"    [diag@iter0, {mode}] z_t arc-to-nearest  "
                          f"mean={arc_to_nearest.mean():.3f}  p95={arc_to_nearest.quantile(0.95):.3f}")

            v_target = v_target + alpha * score

        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            v_pred_raw = model(z_t.detach(), t_global.squeeze(1))
            v_pred = v_pred_raw - (v_pred_raw * z_t).sum(dim=-1, keepdim=True) * z_t
            loss = ((v_pred - v_target.detach()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # LR warmup
        if it < warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr * (it + 1) / warmup_iters
        elif it == warmup_iters:
            for pg in opt.param_groups:
                pg["lr"] = lr
        opt.step()
        # EMA update
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

        if it % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {it:4d}  loss={loss.item():.4f}")

    # Load EMA weights for evaluation
    model.load_state_dict(ema_state)
    return model, losses


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION: GENERATE PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────


def generate_fisher_flow(model, Y0_test, n_steps=50, t_start=0.0, t_end=1.0,
                         score_net=None, alpha=0.0, score_net_sigma=0.1,
                         inf_sigma=0.0):
    """
    Integrate the learned tangent velocity field with Euler steps on the
    sphere, from t_start to t_end. Defaults to the full [0, 1] interval;
    call with t_end < 1.0 for intermediate-timepoint eval.

    When score_net and alpha>0 are supplied, each Euler step is augmented
    with a manifold-correcting score term:

        v_total = v_flow(z, t) + alpha * score(z)

    When inf_sigma > 0, integration switches from deterministic Euler to a
    first-order Euler–Maruyama SDE on the sphere: at each step a tangent-
    space Gaussian increment `sqrt(dt) * inf_sigma * eps_tan` is added on
    top of the drift, giving DDPM-style stochastic sampling. With
    inf_sigma = 0 this reduces exactly to deterministic Euler.
    """
    model.eval()
    use_score = alpha > 0.0 and score_net is not None
    if use_score and isinstance(score_net, nn.Module):
        score_net.eval()
    with torch.no_grad():
        zt = Y0_test.clone().to(DEVICE)
        dt = (t_end - t_start) / n_steps
        log_sigma = float(np.log(score_net_sigma)) if use_score else 0.0
        sqrt_dt = dt ** 0.5
        for step in range(n_steps):
            t_val = torch.full((len(zt),), t_start + step * dt, device=DEVICE)
            v = model(zt, t_val)
            v = v - (v * zt).sum(dim=-1, keepdim=True) * zt
            if use_score:
                log_sig_t = torch.full((len(zt),), log_sigma, device=DEVICE)
                s = score_net(zt, log_sig_t)
                v = v + alpha * s
            if inf_sigma > 0.0:
                eps = torch.randn_like(zt) * inf_sigma
                eps = eps - (eps * zt).sum(dim=-1, keepdim=True) * zt
                zt = normalize_sphere(zt + dt * v + sqrt_dt * eps)
            else:
                zt = normalize_sphere(zt + dt * v)
    return zt.cpu()


def generate_fisher_flow_trajectory(model, Y0_test, n_steps=50, t_start=0.0, t_end=1.0,
                                    n_checkpoints=20):
    """Like generate_fisher_flow but returns intermediate positions for visualization.

    Returns list of (t, positions_cpu) tuples of length n_checkpoints+1
    (including the starting position at t_start).
    """
    model.eval()
    save_every = max(1, n_steps // n_checkpoints)
    with torch.no_grad():
        zt = Y0_test.clone().to(DEVICE)
        dt = (t_end - t_start) / n_steps
        trajectory = [(t_start, from_orthant(zt).cpu())]
        for step in range(n_steps):
            t_val = torch.full((len(zt),), t_start + step * dt, device=DEVICE)
            v = model(zt, t_val)
            v = v - (v * zt).sum(dim=-1, keepdim=True) * zt
            zt = normalize_sphere(zt + dt * v)
            if (step + 1) % save_every == 0 or step == n_steps - 1:
                trajectory.append((t_start + (step + 1) * dt, from_orthant(zt).cpu()))
    return trajectory


def visualize_trajectories_phate(models_dict, test_stage_log1p, test_stage_sphere,
                                 stage_times, stages, n_traj=150, n_steps=100,
                                 out_path="trajectories_phate.html"):
    """Generate PHATE-embedded trajectory visualization as interactive Plotly HTML.

    Fits PHATE on log1p test cells (matching the endpoint visualization and
    the standard PHATE tutorial), integrates each model from t=0 test cells
    to t=1, maps intermediate positions back to log1p space via the
    median-library-size inverse transform, and plots trajectories over the
    empirical cell landscape.
    """
    import phate
    import plotly.graph_objects as go

    S = len(stages)
    all_test_log1p = torch.cat(test_stage_log1p, dim=0)

    print(f"\n  Fitting PHATE on {len(all_test_log1p)} test cells (log1p) for trajectory visualization...")
    ph = phate.PHATE(n_components=2, knn=15, t="auto", verbose=0, random_state=42)
    emb_all = ph.fit_transform(all_test_log1p.numpy())

    all_counts = torch.expm1(all_test_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig = go.Figure()

    # Ground-truth cells as background — grouped for one-click toggle
    offset = 0
    for i, s in enumerate(stages):
        n_s = len(test_stage_log1p[i])
        idx = slice(offset, offset + n_s)
        fig.add_trace(go.Scatter(
            x=emb_all[idx, 0], y=emb_all[idx, 1],
            mode="markers",
            marker=dict(size=4, color=stage_colors[i % len(stage_colors)], opacity=0.3),
            name=f"GT {s} (t={stage_times[i]:.2f})",
            legendgroup="Ground Truth",
            legendgrouptitle_text="Ground Truth" if i == 0 else None,
            showlegend=(i == 0),
        ))
        offset += n_s

    rng = np.random.default_rng(0)
    traj_idx = rng.choice(len(test_stage_sphere[0]), size=min(n_traj, len(test_stage_sphere[0])), replace=False)
    source_subset = test_stage_sphere[0][traj_idx]

    method_colors = {"MM+SLERP": "#000000", "MM+SLERP+Biharmonic": "#e6194b",
                     "MM+SLERP+GlobalBiharmonic": "#ff6600",
                     "MM+SI": "#3cb44b", "MM+SI+Biharmonic": "#4363d8",
                     "MM+Score_learned+Biharmonic": "#f58231"}

    for mname, model in models_dict.items():
        print(f"    Generating {n_traj} trajectories for {mname}...")
        traj = generate_fisher_flow_trajectory(
            model, source_subset, n_steps=n_steps,
            t_start=0.0, t_end=1.0, n_checkpoints=20,
        )

        # Map trajectory positions to log1p space and project through PHATE
        traj_log1p_all = []
        for _, pos_comp in traj:
            traj_log1p_all.append(torch.log1p(pos_comp * median_libsize).numpy())
        traj_log1p_cat = np.concatenate(traj_log1p_all, axis=0)
        traj_emb = ph.transform(traj_log1p_cat)

        n_cells = len(source_subset)
        n_ckpt = len(traj)
        color = method_colors.get(mname, "#888888")

        for ci in range(n_cells):
            xs = [traj_emb[k * n_cells + ci, 0] for k in range(n_ckpt)]
            ys = [traj_emb[k * n_cells + ci, 1] for k in range(n_ckpt)]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=0.5),
                opacity=0.4,
                showlegend=(ci == 0),
                name=mname,
                legendgroup=mname,
                legendgrouptitle_text=mname if ci == 0 else None,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title="Predicted Trajectories on PHATE Embedding (log1p space)",
        xaxis_title="PHATE 1", yaxis_title="PHATE 2",
        width=1100, height=850,
        legend=dict(font=dict(size=11), groupclick="togglegroup"),
    )
    fig.write_html(out_path)
    print(f"    Saved interactive visualization to {out_path}")


def visualize_endpoints_phate(models_dict, test_stage_log1p, test_stage_sphere,
                              stage_times, stages, out_path="endpoints_phate.html"):
    """PHATE scatter of ground-truth vs predicted cell distributions at each stage.

    PHATE is fitted on the raw log1p-normalized test cells (matching the
    standard PHATE tutorial pipeline). Flow predictions are mapped from
    the sphere back to log1p space via:
        sphere → from_orthant (y²) → compositional → * median_libsize → log1p
    then projected through phate.transform().

    Legend is grouped so clicking "Ground Truth" toggles all GT stages at once,
    and clicking a method name toggles all that method's predicted stages.
    """
    import phate
    import plotly.graph_objects as go

    S = len(stages)

    # Fit PHATE on raw log1p test cells (matching the standard tutorial)
    all_test_log1p = torch.cat(test_stage_log1p, dim=0)
    print(f"\n  Fitting PHATE on {len(all_test_log1p)} test cells (log1p space)...")
    ph = phate.PHATE(n_components=2, knn=15, t="auto", verbose=0, random_state=42)
    emb_all = ph.fit_transform(all_test_log1p.numpy())

    # Compute median library size from all test cells for inverse transform
    all_counts = torch.expm1(all_test_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())
    print(f"    Median library size (test cells): {median_libsize:.1f}")

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    fig = go.Figure()

    # Ground-truth cells per stage — all under one legendgroup
    offset = 0
    for i, s in enumerate(stages):
        n_s = len(test_stage_log1p[i])
        idx = slice(offset, offset + n_s)
        fig.add_trace(go.Scatter(
            x=emb_all[idx, 0], y=emb_all[idx, 1],
            mode="markers",
            marker=dict(size=5, color=stage_colors[i % len(stage_colors)], opacity=0.4),
            name=f"GT {s} (t={stage_times[i]:.2f})",
            legendgroup="Ground Truth",
            legendgrouptitle_text="Ground Truth" if i == 0 else None,
            showlegend=(i == 0),
        ))
        offset += n_s

    method_markers = {"MM+SLERP": "diamond", "MM+SLERP+Biharmonic": "x",
                      "MM+SI": "cross", "MM+SI+Biharmonic": "star",
                      "MM+Score_learned+Biharmonic": "triangle-up"}

    source = test_stage_sphere[0]
    for mname, model in models_dict.items():
        print(f"    Generating chained predictions for {mname}...")
        marker_sym = method_markers.get(mname, "circle")
        for i in range(1, S):
            t_target = stage_times[i]
            n_steps = max(10, int(50 * t_target))
            pred_sphere = generate_fisher_flow(
                model, source, n_steps=n_steps, t_start=0.0, t_end=t_target,
            )
            # Map predictions back to log1p space: sphere → comp → counts → log1p
            pred_comp = from_orthant(pred_sphere)
            pred_counts = pred_comp * median_libsize
            pred_log1p = torch.log1p(pred_counts).numpy()
            pred_emb = ph.transform(pred_log1p)
            fig.add_trace(go.Scatter(
                x=pred_emb[:, 0], y=pred_emb[:, 1],
                mode="markers",
                marker=dict(size=7, symbol=marker_sym, opacity=0.7,
                            color=stage_colors[i % len(stage_colors)],
                            line=dict(width=0.5, color="black")),
                name=f"{mname} → t={t_target:.2f}",
                legendgroup=mname,
                legendgrouptitle_text=mname if i == 1 else None,
                showlegend=(i == 1),
            ))

    fig.update_layout(
        title="Ground Truth vs Predicted Endpoints (PHATE, log1p space)",
        xaxis_title="PHATE 1", yaxis_title="PHATE 2",
        width=1100, height=850,
        legend=dict(font=dict(size=11), groupclick="togglegroup"),
    )
    fig.write_html(out_path)
    print(f"    Saved endpoint visualization to {out_path}")


def visualize_nn_distance_histograms(models_dict, test_stage_log1p, test_stage_sphere,
                                     stage_times, stages, out_path="nn_distances.html"):
    """Nearest-neighbor distance histograms: predicted → ground-truth per stage.

    For each predicted cell at stage i, computes its Euclidean distance (in
    log1p space) to the closest ground-truth cell at that stage. Overlapping
    histograms for each method let you compare how well predictions match the
    true cell distribution.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    S = len(stages)

    # Compute median library size for inverse transform
    all_log1p = torch.cat(test_stage_log1p, dim=0)
    all_counts = torch.expm1(all_log1p).clamp(min=0)
    median_libsize = float(all_counts.sum(dim=-1).median())

    method_colors = {"MM+SLERP": "rgba(0,0,0,0.5)", "MM+SLERP+Biharmonic": "rgba(230,25,75,0.5)",
                     "MM+SI": "rgba(60,180,75,0.5)", "MM+SI+Biharmonic": "rgba(67,99,216,0.5)",
                     "MM+Score_learned+Biharmonic": "rgba(245,130,49,0.5)"}

    fig = make_subplots(
        rows=1, cols=S - 1,
        subplot_titles=[f"t={stage_times[i]:.2f} ({stages[i]})" for i in range(1, S)],
        shared_yaxes=True,
    )

    source = test_stage_sphere[0]
    for mname, model in models_dict.items():
        print(f"    Computing NN distances for {mname}...")
        color = method_colors.get(mname, "rgba(128,128,128,0.5)")
        for col_idx, i in enumerate(range(1, S)):
            t_target = stage_times[i]
            n_steps = max(10, int(50 * t_target))
            pred_sphere = generate_fisher_flow(
                model, source, n_steps=n_steps, t_start=0.0, t_end=t_target,
            )
            pred_comp = from_orthant(pred_sphere)
            pred_log1p = torch.log1p(pred_comp * median_libsize)
            gt_log1p = test_stage_log1p[i]

            dists = torch.cdist(pred_log1p, gt_log1p)
            nn_dists = dists.min(dim=1).values.numpy()

            fig.add_trace(
                go.Histogram(
                    x=nn_dists, nbinsx=50,
                    name=mname,
                    marker_color=color,
                    legendgroup=mname,
                    showlegend=(col_idx == 0),
                ),
                row=1, col=col_idx + 1,
            )

    fig.update_layout(
        title="Nearest-Neighbor Distance to Ground Truth (log1p Euclidean)",
        barmode="overlay",
        width=300 * (S - 1) + 100, height=400,
        legend=dict(font=dict(size=11)),
    )
    for col_idx in range(S - 1):
        fig.update_xaxes(title_text="NN distance" if col_idx == 0 else "", row=1, col=col_idx + 1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.write_html(out_path)
    print(f"    Saved NN distance histograms to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────────────────────────────────────


def mmd_rbf(X, Y, sigma=None):
    """MMD² with RBF kernel. X, Y: (n, D) tensors."""
    if sigma is None:
        # Median heuristic
        all_pts = torch.cat([X, Y], dim=0)
        dists = torch.cdist(all_pts, all_pts)
        sigma = dists.median().item() / np.sqrt(2)
        sigma = max(sigma, 1e-3)

    def k(A, B):
        d = torch.cdist(A, B) ** 2
        return torch.exp(-d / (2 * sigma**2))

    n, m = len(X), len(Y)
    kxx = k(X, X)
    kyy = k(Y, Y)
    kxy = k(X, Y)
    mmd2 = (
        (kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean()
    )
    return mmd2.item()


def cosine_logfc(pred_p, true_p, ctrl_p):
    """
    Cosine similarity between predicted and true log fold-change vectors.
    pred_p, true_p, ctrl_p: (n, D) compositional vectors
    """
    ctrl_mean = ctrl_p.mean(dim=0).clamp(min=1e-8)
    pred_logfc = torch.log(pred_p.mean(dim=0).clamp(min=1e-8) / ctrl_mean)
    true_logfc = torch.log(true_p.mean(dim=0).clamp(min=1e-8) / ctrl_mean)
    cos = torch.nn.functional.cosine_similarity(pred_logfc.unsqueeze(0), true_logfc.unsqueeze(0)).item()
    return cos


# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────


def plot_results(ctrl_p, pert_p, preds_dict, title="Results"):
    """
    PCA visualization of predictions vs ground truth in simplex space.
    """
    from sklearn.decomposition import PCA

    all_data = np.vstack([ctrl_p.cpu().numpy(), pert_p.cpu().numpy(), *[v.cpu().numpy() for v in preds_dict.values()]])
    pca = PCA(n_components=2)
    pca.fit(all_data)

    fig, axes = plt.subplots(1, len(preds_dict) + 2, figsize=(4 * (len(preds_dict) + 2), 4))

    def scatter(ax, data, label, color, alpha=0.5):
        xy = pca.transform(data.cpu().numpy())
        ax.scatter(xy[:, 0], xy[:, 1], c=color, alpha=alpha, s=15, label=label)

    # Control + true perturbed
    for ax in axes:
        scatter(ax, ctrl_p, "control", "#888", alpha=0.3)
        scatter(ax, pert_p, "true perturbed", "#e24b4a", alpha=0.4)

    axes[0].set_title("Control vs True")
    axes[1].set_title("Control vs True (ref)")

    for i, (name, pred) in enumerate(preds_dict.items()):
        ax = axes[i + 2]
        scatter(ax, pred, name, "#378add", alpha=0.6)
        ax.set_title(name)

    for ax in axes:
        ax.legend(fontsize=7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("orthant_pmf_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot to orthant_pmf_results.png")


SCORE_ALPHAS = (0.05, 0.1, 0.2, 0.5)

METHOD_NAMES = [
    "Sphere+SLERP", "Graph+SLERP", "Sphere+Polyline", "Graph+Polyline", "Graph+Hybrid",
    "Graph+Anchor1", "Graph+Anchor2", "Graph+Anchor3", "Graph+Anchor5",
] + [f"Sphere+Score{a}" for a in SCORE_ALPHAS]
METHOD_COLORS = {
    "Sphere+SLERP": "#888888",     # vanilla Fisher Flow baseline (gray)
    "Graph+SLERP": "#f5a623",      # cost-only swap (orange)
    "Sphere+Polyline": "#378add",  # path-only swap (blue)
    "Graph+Polyline": "#1d9e75",   # full polyline, consistent (green)
    "Graph+Hybrid": "#c73e9a",     # polyline positions + SLERP velocities (magenta)
    "Graph+Anchor1": "#2e86c1",    # one-anchor sparse polyline
    "Graph+Anchor2": "#884ea0",    # two-anchor sparse polyline
    "Graph+Anchor3": "#a93226",    # three-anchor sparse polyline
    "Graph+Anchor5": "#d35400",    # five-anchor sparse polyline
    "Sphere+Score0.05": "#76d7c4", # score reg α=0.05 (faint teal)
    "Sphere+Score0.1":  "#1abc9c", # score reg α=0.1 (teal)
    "Sphere+Score0.2":  "#148f77", # score reg α=0.2 (deep teal)
    "Sphere+Score0.5":  "#0b5345", # score reg α=0.5 (very deep teal)
}


def plot_ot_coupling_comparison(cost_sphere, cost_graph, Y0, Y1, n_show=30, filename="ot_coupling_comparison.png"):
    """Visualize how OT pairings differ between sphere-cost and graph shortest-path cost."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    all_pts = np.vstack([Y0.cpu().numpy(), Y1.cpu().numpy()])
    pca.fit(all_pts)
    xy_src = pca.transform(Y0.cpu().numpy())
    xy_tgt = pca.transform(Y1.cpu().numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, cost, title in [(ax1, cost_sphere, "Sphere arc-length OT"), (ax2, cost_graph, "Graph shortest-path OT")]:
        src, tgt = ot_coupling(cost, n_show)
        ax.scatter(xy_src[:, 0], xy_src[:, 1], c="#888", s=20, alpha=0.4, label="source")
        ax.scatter(xy_tgt[:, 0], xy_tgt[:, 1], c="#e24b4a", s=20, alpha=0.4, label="target")
        for s, t in zip(src[:n_show], tgt[:n_show]):
            ax.plot(
                [xy_src[s, 0], xy_tgt[t, 0]],
                [xy_src[s, 1], xy_tgt[t, 1]],
                c="#378add", alpha=0.3, lw=0.8,
            )
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved OT coupling comparison to {filename}")


def plot_loss_curves(losses_dict, filename="loss_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, color=METHOD_COLORS.get(name, "#888"), lw=2)
    ax.set_xlabel("Checkpoint (every 200 iters)")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. EMBRYOID BODY DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────


def load_embryoid_body(path="embryoid_body.h5ad", n_hvg=2000, val_frac=0.1, test_frac=0.1, seed=42, max_cells_per_stage=None):
    """
    Load embryoid body dataset with 5 timepoints and split into train/val/test.

    The dataset has stages: 0-1, 2-3, 4-5, 6-7, 8-9.
    We treat consecutive timepoints as (source, target) pairs for flow matching:
      (0-1 -> 2-3), (2-3 -> 4-5), (4-5 -> 6-7), (6-7 -> 8-9)

    Split: 10% test, 10% val, 80% train — stratified per timepoint so each
    split has proportional representation from every stage.

    Returns dict with keys 'train', 'val', 'test', each containing:
      - 'stages': ordered list of stage labels
      - 'cells': dict mapping stage -> torch.Tensor of shape (n_cells, n_hvg)
      - 'cell_types': dict mapping stage -> np.array of cell type labels
      - 'transitions': list of (src_stage, tgt_stage) pairs for flow matching
    """
    import anndata as ad
    import scipy.sparse

    print(f"Loading {path}...")
    adata = ad.read_h5ad(path)
    print(f"  Raw: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Subset to highly variable genes if available, otherwise select top HVG
    if "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values
        n_hvg_available = hvg_mask.sum()
        if n_hvg_available >= n_hvg:
            # Take top n_hvg by dispersion norm among HVGs
            hvg_idx = np.where(hvg_mask)[0]
            disp = adata.var["dispersions_norm"].values[hvg_idx]
            top_idx = hvg_idx[np.argsort(disp)[::-1][:n_hvg]]
            adata = adata[:, top_idx]
        else:
            adata = adata[:, hvg_mask]
        print(f"  After HVG filter: {adata.shape[1]} genes")
    else:
        print(f"  No HVG annotation found, using all {adata.shape[1]} genes")

    # Ordered stages
    stages = ["0-1", "2-3", "4-5", "6-7", "8-9"]
    transitions = [(stages[i], stages[i + 1]) for i in range(len(stages) - 1)]

    # Stratified split per timepoint
    rng = np.random.default_rng(seed)
    split_data = {s: {"stages": stages, "cells": {}, "cell_types": {}, "transitions": transitions} for s in ["train", "val", "test"]}

    for stage in stages:
        mask = adata.obs["stage"] == stage
        idx = np.where(mask)[0]
        rng.shuffle(idx)

        # Subsample if requested
        if max_cells_per_stage is not None and len(idx) > max_cells_per_stage:
            idx = idx[:max_cells_per_stage]

        n = len(idx)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]

        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            X = adata.X[split_idx]
            if scipy.sparse.issparse(X):
                X = X.toarray()
            split_data[split_name]["cells"][stage] = torch.tensor(X, dtype=torch.float32)
            split_data[split_name]["cell_types"][stage] = adata.obs["cell_type"].values[split_idx]

    # Print summary
    print(f"\n  Split summary (val={val_frac:.0%}, test={test_frac:.0%}):")
    print(f"  {'Stage':<8} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*31}")
    for stage in stages:
        nt = len(split_data["train"]["cells"][stage])
        nv = len(split_data["val"]["cells"][stage])
        ne = len(split_data["test"]["cells"][stage])
        print(f"  {stage:<8} {nt:>7} {nv:>7} {ne:>7}")
    total_tr = sum(len(split_data["train"]["cells"][s]) for s in stages)
    total_va = sum(len(split_data["val"]["cells"][s]) for s in stages)
    total_te = sum(len(split_data["test"]["cells"][s]) for s in stages)
    print(f"  {'Total':<8} {total_tr:>7} {total_va:>7} {total_te:>7}")
    print(f"\n  Transitions: {transitions}")
    print(f"  Gene dim: {adata.shape[1]}")

    return split_data


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────


def build_ablation(Y0, Y1, k=15, score_cells=None, sigma=None):
    """
    Build the full ablation grid on the positive orthant. Shares the
    PolylineGeodesic (and its underlying kNN graph + Dijkstra) across the
    cells that use it, so the graph is only built once per transition.

    When score_cells and sigma are provided, adds Sphere+Score{alpha} methods
    for each alpha in SCORE_ALPHAS. These use the vanilla SLERP interpolant
    but augment the training velocity target with alpha * ∇log p(z_t) where
    p is an RBF-KDE over score_cells.

    Returns a dict: method_name -> (interpolant, cost_matrix, train_kwargs).
    train_kwargs is passed as **kwargs to train_fisher_flow, empty for
    non-score methods.
    """
    slerp_interp = SLERPInterpolant(Y0, Y1)
    poly_interp = PolylineGeodesic(Y0, Y1, k=k)
    hybrid_interp = PolylinePosSLERPVelInterpolant(poly_interp, slerp_interp)
    anchor_interps = {n: SparseAnchorGeodesic(poly_interp, n_anchors=n) for n in (1, 2, 3, 5)}

    cost_sphere = compute_sphere_cost_matrix(Y0, Y1)
    cost_graph = poly_interp.cost_matrix

    ablation = {
        "Sphere+SLERP":    (slerp_interp,       cost_sphere, {}),
        "Graph+SLERP":     (slerp_interp,       cost_graph,  {}),
        "Sphere+Polyline": (poly_interp,        cost_sphere, {}),
        "Graph+Polyline":  (poly_interp,        cost_graph,  {}),
        "Graph+Hybrid":    (hybrid_interp,      cost_graph,  {}),
        "Graph+Anchor1":   (anchor_interps[1],  cost_graph,  {}),
        "Graph+Anchor2":   (anchor_interps[2],  cost_graph,  {}),
        "Graph+Anchor3":   (anchor_interps[3],  cost_graph,  {}),
        "Graph+Anchor5":   (anchor_interps[5],  cost_graph,  {}),
    }
    if score_cells is not None and sigma is not None:
        for a in SCORE_ALPHAS:
            ablation[f"Sphere+Score{a}"] = (
                slerp_interp,
                cost_sphere,
                {"score_cells": score_cells, "sigma": sigma, "alpha": float(a)},
            )

    return ablation, cost_sphere, cost_graph, poly_interp


def train_ablation(ablation, D, n_iters, batch_size):
    """Train every method in METHOD_NAMES that has an entry in the ablation dict.

    Entries are 3-tuples (interpolant, cost_matrix, train_kwargs).
    train_kwargs is splatted into train_fisher_flow, enabling per-method knobs
    like the score regularizer without touching the training loop signature.
    """
    models = {}
    losses = {}
    for name in METHOD_NAMES:
        if name not in ablation:
            continue
        interp, cost, extra = ablation[name]
        print(f"\n  Training {name}...")
        models[name], losses[name] = train_fisher_flow(
            interp, cost, D, n_iters=n_iters, batch_size=batch_size, label=name, **extra,
        )
    return models, losses


def main_toy():
    """
    Toy 2x2 ablation on swiss-roll compositional data. Fast sanity check
    that the data-faithful Fisher FM pipeline runs end to end before
    touching the embryoid body dataset.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    D = 32
    N_TRAIN = 400
    N_TEST = 200
    N_ITERS = 2000
    BATCH = 256

    print("=" * 60)
    print("Generating compositional toy data...")
    X_ctrl_train, X_pert_train = make_compositional_data(N_TRAIN, D, seed=42)
    X_ctrl_test, X_pert_test = make_compositional_data(N_TEST, D, seed=99)

    Y0_train = normalize_sphere(to_orthant(X_ctrl_train)).to(DEVICE)
    Y1_train = normalize_sphere(to_orthant(X_pert_train)).to(DEVICE)
    Y0_test = normalize_sphere(to_orthant(X_ctrl_test)).to(DEVICE)

    print(f"Data: {N_TRAIN} train cells, {N_TEST} test cells, D={D}")
    print(f"Y0 norm check: {Y0_train.norm(dim=-1).mean():.4f} (should be ~1.0)")

    print("\nBuilding ablation grid (kNN graph + Dijkstra shared across variants)...")
    ablation, cost_sphere, cost_graph, _ = build_ablation(Y0_train, Y1_train, k=15)
    print(f"  Sphere cost range: [{cost_sphere.min():.3f}, {cost_sphere.max():.3f}]")
    print(f"  Graph  cost range: [{cost_graph.min():.3f}, {cost_graph.max():.3f}]")

    plot_ot_coupling_comparison(cost_sphere, cost_graph, Y0_train, Y1_train)

    print("\n" + "=" * 60)
    models, losses = train_ablation(ablation, D, N_ITERS, BATCH)
    plot_loss_curves(losses)

    print("\n" + "=" * 60)
    print("Generating predictions on test set...")
    preds_simplex = {}
    for name, model in models.items():
        pred_y = generate_fisher_flow(model, Y0_test, n_steps=50)
        preds_simplex[name] = from_orthant(pred_y)

    print("\n" + "=" * 60)
    print("Evaluation metrics on test set:")
    print(f"  {'Model':<20} {'MMD²':>10} {'Cos logFC':>12}")
    print("  " + "-" * 44)
    for name in METHOD_NAMES:
        pred_p = preds_simplex[name]
        mmd = mmd_rbf(pred_p, X_pert_test)
        cos = cosine_logfc(pred_p, X_pert_test, X_ctrl_test)
        print(f"  {name:<20} {mmd:>10.4f} {cos:>12.4f}")

    plot_results(
        X_ctrl_test,
        X_pert_test,
        preds_simplex,
        title="Toy predictions in PCA space (test set)",
    )
    print("\nDone.")


STAGE_COLORS = {
    "0-1": "#636EFA", "2-3": "#EF553B", "4-5": "#00CC96",
    "6-7": "#AB63FA", "8-9": "#FFA15A",
}


def plot_embryoid_loo_results(split_data, loo_results, stages, filename="embryoid_results.html"):
    """
    Leave-one-timepoint-out visualization. For each LOO transition
    (src, held, tgt), show the training PHATE manifold and overlay:
      - true source test cells (circles, src color)
      - true held-out test cells (circles, held color, bigger — the target)
      - true target test cells (circles, tgt color)
      - intermediate prediction (diamonds, held color) — should overlap circles
    One row per LOO transition, one column per method + ground truth.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # ── Fit PHATE on all training cells (dense manifold) ──────────────────
    print("  Fitting PHATE on training cells for visualization...")
    train_arrays = []
    for s in stages:
        P_train = to_compositional(split_data["train"]["cells"][s]).cpu().numpy()
        train_arrays.append(P_train)
    all_train = np.vstack(train_arrays)

    if HAS_PHATE:
        phate_op = phate.PHATE(n_components=2, knn=15, t="auto", verbose=0, random_state=42)
        train_embedding = phate_op.fit_transform(all_train)
    else:
        from sklearn.decomposition import PCA
        print("  PHATE not available, using PCA...")
        phate_op = PCA(n_components=2).fit(all_train)
        train_embedding = phate_op.transform(all_train)

    train_embed_by_stage = {}
    offset = 0
    for i, s in enumerate(stages):
        n = len(train_arrays[i])
        train_embed_by_stage[s] = train_embedding[offset : offset + n]
        offset += n

    # ── Build subplot grid: rows = LOO transitions, cols = GT + methods ───
    loo_keys = list(loo_results.keys())
    n_rows = len(loo_keys)
    n_cols = 1 + len(METHOD_NAMES)
    col_titles = ["Ground Truth"] + METHOD_NAMES
    subplot_titles = []
    for src, held, tgt in loo_keys:
        for c in range(n_cols):
            prefix = f"{src}→{held}→{tgt}  " if c == 0 else ""
            subplot_titles.append(f"{prefix}{col_titles[c]}")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03, vertical_spacing=0.08,
    )

    show_stage_legend = True
    for row, key in enumerate(loo_keys, 1):
        src_stage, held_stage, tgt_stage = key
        res = loo_results[key]

        # Project test cells + intermediate predictions
        P_src_test = res["P_src_test"].cpu().numpy()
        P_held_test = res["P_held_test"].cpu().numpy()
        P_tgt_test = res["P_tgt_test"].cpu().numpy()
        embed_src = phate_op.transform(P_src_test)
        embed_held = phate_op.transform(P_held_test)
        embed_tgt = phate_op.transform(P_tgt_test)

        for col in range(1, n_cols + 1):
            # Faint training manifold as background (all stages)
            for s in stages:
                xy = train_embed_by_stage[s]
                fig.add_trace(
                    go.Scatter(
                        x=xy[:, 0], y=xy[:, 1], mode="markers",
                        marker=dict(color=STAGE_COLORS[s], size=3, opacity=0.1),
                        showlegend=False, hoverinfo="skip",
                    ),
                    row=row, col=col,
                )

            # True test cells for the three relevant stages
            for label, xy, stage, size in [
                ("src", embed_src, src_stage, 5),
                ("held", embed_held, held_stage, 6),
                ("tgt", embed_tgt, tgt_stage, 5),
            ]:
                fig.add_trace(
                    go.Scatter(
                        x=xy[:, 0], y=xy[:, 1], mode="markers",
                        marker=dict(color=STAGE_COLORS[stage], size=size, opacity=0.7,
                                    line=dict(width=0.5, color="white")),
                        name=f"Stage {stage}" + (" (held)" if label == "held" else ""),
                        legendgroup=f"stage_{stage}",
                        showlegend=show_stage_legend,
                    ),
                    row=row, col=col,
                )

            # Predictions (intermediate at t=0.5) as diamonds, only in method columns
            if col >= 2:
                method = METHOD_NAMES[col - 2]
                pred_mid = res["preds_intermediate"][method].cpu().numpy()
                pred_xy = phate_op.transform(pred_mid)
                fig.add_trace(
                    go.Scatter(
                        x=pred_xy[:, 0], y=pred_xy[:, 1], mode="markers",
                        marker=dict(
                            color=STAGE_COLORS[held_stage], size=7, opacity=0.75,
                            symbol="diamond",
                            line=dict(width=0.7, color="black"),
                        ),
                        name=f"{method} → t=0.5",
                        legendgroup=f"pred_{method}",
                        showlegend=(row == 1 and col == 2),
                    ),
                    row=row, col=col,
                )
            show_stage_legend = False  # only show legend once

    fig.update_layout(
        title="Embryoid Body Leave-One-Timepoint-Out: intermediate predictions at t=0.5 vs held-out stage",
        height=max(500, 380 * n_rows), width=340 * n_cols,
        template="plotly_white",
        legend=dict(font=dict(size=10)),
    )
    fig.update_xaxes(title_text="PHATE-1")
    fig.update_yaxes(title_text="PHATE-2")
    fig.write_html(filename)
    print(f"Saved {filename}")


def plot_embryoid_loss_curves(loo_losses, filename="embryoid_loss_curves.html"):
    """Loss curves per LOO transition, coloured by method."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    loo_keys = list(loo_losses.keys())
    fig = make_subplots(
        rows=1, cols=len(loo_keys),
        subplot_titles=[f"{s}→{h}→{t}" for s, h, t in loo_keys],
        horizontal_spacing=0.05,
    )
    for col, key in enumerate(loo_keys, 1):
        for method in METHOD_NAMES:
            losses = loo_losses[key].get(method)
            if losses is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(losses))),
                    y=losses,
                    mode="lines",
                    name=method,
                    legendgroup=method,
                    showlegend=(col == 1),
                    line=dict(color=METHOD_COLORS.get(method, "#888"), width=2),
                ),
                row=1, col=col,
            )
    fig.update_xaxes(title_text="Checkpoint (every 200 iters)")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(
        title="Training loss curves (EB leave-one-out)",
        height=400, width=360 * len(loo_keys),
        template="plotly_white",
    )
    fig.write_html(filename)
    print(f"Saved {filename}")


def _subsample_tensor(X, n, rng):
    """Random subsample rows of a torch tensor."""
    if len(X) <= n:
        return X, np.arange(len(X))
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], idx


def main_diagnose(ot_subsample=2000, knn=15, n_show_paths=30):
    """
    Diagnostic report for the polyline interpolant on the first LOO transition.
    Tests why polyline underperforms SLERP. Reports:
      - path length histogram (len=2 ≡ SLERP degenerate)
      - polyline / SLERP arc length ratio per pair
      - interior waypoint reuse + direction conflict (multi-valued field test)
      - velocity magnitude distribution (SLERP vs polyline)
      - PHATE overlay of n_show_paths sampled polylines vs SLERP chords
    Writes polyline_diagnostic.html.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from collections import defaultdict

    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    data = load_embryoid_body("embryoid_body.h5ad", n_hvg=2000)
    stages = data["train"]["stages"]

    src_stage, held_stage, tgt_stage = stages[0], stages[1], stages[2]
    print(f"\n{'=' * 60}")
    print(f"DIAGNOSE  {src_stage} -> [{held_stage}] -> {tgt_stage}")
    print("=" * 60)

    P_src = to_compositional(data["train"]["cells"][src_stage])
    P_tgt = to_compositional(data["train"]["cells"][tgt_stage])
    P_held = to_compositional(data["train"]["cells"][held_stage])

    Y0 = normalize_sphere(to_orthant(P_src)).to(DEVICE)
    Y1 = normalize_sphere(to_orthant(P_tgt)).to(DEVICE)
    Y_held = normalize_sphere(to_orthant(P_held)).to(DEVICE)

    Y0_sub, _ = _subsample_tensor(Y0, ot_subsample, rng)
    Y1_sub, _ = _subsample_tensor(Y1, ot_subsample, rng)
    Y_held_sub, _ = _subsample_tensor(Y_held, min(1000, len(Y_held)), rng)

    print(f"\n  Building polyline (n={len(Y0_sub)}+{len(Y1_sub)}, k={knn})...")
    poly = PolylineGeodesic(Y0_sub, Y1_sub, k=knn)
    slerp = SLERPInterpolant(Y0_sub, Y1_sub)
    cost_sphere = compute_sphere_cost_matrix(Y0_sub, Y1_sub)
    cost_graph = poly.cost_matrix

    # OT pairs via graph cost — the coupling actually used by Graph+Polyline
    n_pool = 5000
    print(f"  Sampling {n_pool} OT pairs via graph cost...")
    ot_src, ot_tgt = ot_coupling(cost_graph, n_pool)

    # Reconstruct all paths
    paths = [poly.reconstruct_path(int(s), int(g)) for s, g in zip(ot_src, ot_tgt)]
    path_lens = np.array([len(p) for p in paths])

    # ── Metric 1: path lengths ────────────────────────────────────────────
    print(f"\n  [1] Path length (nodes):")
    print(f"      min={path_lens.min()}  max={path_lens.max()}  mean={path_lens.mean():.2f}  median={int(np.median(path_lens))}")
    print(f"      %len=2 (degenerate ≡ SLERP): {(path_lens == 2).mean() * 100:.1f}%")
    print(f"      %len>=5 (real curvature):    {(path_lens >= 5).mean() * 100:.1f}%")

    # ── Metric 2: arc length ratio polyline / SLERP ───────────────────────
    ratio = cost_graph[ot_src, ot_tgt] / cost_sphere[ot_src, ot_tgt].clip(1e-8)
    print(f"\n  [2] Polyline / SLERP arc length ratio:")
    print(f"      min={ratio.min():.2f}  mean={ratio.mean():.2f}  p95={np.percentile(ratio, 95):.2f}  max={ratio.max():.2f}")

    # ── Metric 3: waypoint reuse + direction conflict ─────────────────────
    # For every node seen as an interior/source waypoint, count outgoing
    # edges across all OT pairs. Multi-valued → multi-valued target field.
    next_map = defaultdict(lambda: defaultdict(int))
    for p in paths:
        for i in range(len(p) - 1):
            next_map[p[i]][p[i + 1]] += 1

    reused_nodes = [(node, nexts) for node, nexts in next_map.items() if sum(nexts.values()) >= 2]
    passes = np.array([sum(nxt.values()) for _, nxt in reused_nodes])
    distinct = np.array([len(nxt) for _, nxt in reused_nodes])
    # Herfindahl-style concentration: sum(p_i^2); 1=single direction, small=spread
    concentration = np.array([
        sum((c / sum(nxt.values())) ** 2 for c in nxt.values()) for _, nxt in reused_nodes
    ]) if reused_nodes else np.array([])
    print(f"\n  [3] Waypoint reuse (nodes used by ≥2 OT pairs):")
    print(f"      reused nodes: {len(reused_nodes)} of {len(next_map)} distinct path-nodes")
    if len(reused_nodes):
        print(f"      passes/node:   mean={passes.mean():.1f}  p95={np.percentile(passes, 95):.0f}  max={passes.max()}")
        print(f"      distinct next: mean={distinct.mean():.2f}  max={distinct.max()}")
        print(f"      % with >1 outgoing direction: {(distinct > 1).mean() * 100:.1f}%  ← conflict rate")
        print(f"      direction concentration (1=pure, <1=mixed): mean={concentration.mean():.3f}  median={np.median(concentration):.3f}")

    # ── Metric 4: velocity magnitudes ─────────────────────────────────────
    n_sample = min(2000, len(ot_src))
    s_idx = rng.choice(len(ot_src), size=n_sample, replace=False)
    t_sample = torch.rand(n_sample, 1, device=DEVICE)
    with torch.no_grad():
        _, v_s = slerp.sample(ot_src[s_idx], ot_tgt[s_idx], t_sample)
        _, v_p = poly.sample(ot_src[s_idx], ot_tgt[s_idx], t_sample)
    mag_s = v_s.norm(dim=-1).cpu().numpy()
    mag_p = v_p.norm(dim=-1).cpu().numpy()
    print(f"\n  [4] Velocity magnitudes ||v_t||:")
    print(f"      SLERP:    mean={mag_s.mean():.3f}  std={mag_s.std():.3f}  max={mag_s.max():.3f}")
    print(f"      Polyline: mean={mag_p.mean():.3f}  std={mag_p.std():.3f}  max={mag_p.max():.3f}")
    print(f"      ratio poly/slerp (mean): {mag_p.mean() / mag_s.mean():.2f}")

    # ── Visualization: PHATE overlay ──────────────────────────────────────
    print(f"\n  [5] Embedding + path overlay...")
    all_pts = torch.cat([Y0_sub.cpu(), Y1_sub.cpu(), Y_held_sub.cpu()], dim=0).numpy()
    if HAS_PHATE:
        try:
            ph = phate.PHATE(n_components=2, n_jobs=-1, verbose=False, random_state=42)
            emb = ph.fit_transform(all_pts)
            embed_name = "PHATE"
        except Exception as e:
            print(f"      PHATE failed ({e}); using PCA")
            from sklearn.decomposition import PCA
            emb = PCA(n_components=2).fit_transform(all_pts)
            embed_name = "PCA"
    else:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(all_pts)
        embed_name = "PCA"

    n0, n1 = len(Y0_sub), len(Y1_sub)
    emb_y0 = emb[:n0]
    emb_y1 = emb[n0 : n0 + n1]
    emb_held = emb[n0 + n1 :]
    node_emb = np.vstack([emb_y0, emb_y1])  # indexable by poly.all_nodes row

    # Prefer longer paths so the visualization isn't all length-2 degenerates
    long_order = np.argsort(-path_lens)
    candidate = long_order[: max(100, n_show_paths * 3)]
    show_idx = rng.choice(candidate, size=min(n_show_paths, len(candidate)), replace=False)

    # ── Plotly figure: 2x2 (overlay, path-len hist, ratio hist, mag hist) ─
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"{embed_name}: polyline (black) vs SLERP chord (orange)  [{src_stage}→{tgt_stage}]",
            "Path length histogram",
            "Polyline / SLERP arc length ratio",
            "Velocity magnitude ||v_t||",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    # Scatter: src / tgt / held
    fig.add_trace(go.Scatter(
        x=emb_y0[:, 0], y=emb_y0[:, 1], mode="markers",
        marker=dict(size=4, color="steelblue", opacity=0.5),
        name=f"src ({src_stage})",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=emb_y1[:, 0], y=emb_y1[:, 1], mode="markers",
        marker=dict(size=4, color="indianred", opacity=0.5),
        name=f"tgt ({tgt_stage})",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=emb_held[:, 0], y=emb_held[:, 1], mode="markers",
        marker=dict(size=4, color="mediumseagreen", opacity=0.5),
        name=f"held ({held_stage})",
    ), row=1, col=1)

    # Polylines and SLERP chords for sampled pairs
    for i, idx in enumerate(show_idx):
        p = paths[idx]
        pts = node_emb[p]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="lines+markers",
            line=dict(color="black", width=1),
            marker=dict(size=3, color="black"),
            name="polyline" if i == 0 else None,
            showlegend=(i == 0), legendgroup="polyline",
        ), row=1, col=1)
        sp, ep = node_emb[p[0]], node_emb[p[-1]]
        fig.add_trace(go.Scatter(
            x=[sp[0], ep[0]], y=[sp[1], ep[1]], mode="lines",
            line=dict(color="orange", width=1, dash="dot"),
            name="SLERP chord" if i == 0 else None,
            showlegend=(i == 0), legendgroup="slerp",
        ), row=1, col=1)

    # Histograms
    fig.add_trace(go.Histogram(x=path_lens, nbinsx=40, marker_color="steelblue", showlegend=False), row=1, col=2)
    fig.add_trace(go.Histogram(x=ratio, nbinsx=40, marker_color="purple", showlegend=False), row=2, col=1)
    fig.add_trace(go.Histogram(x=mag_s, nbinsx=40, marker_color="orange", opacity=0.6, name="SLERP"), row=2, col=2)
    fig.add_trace(go.Histogram(x=mag_p, nbinsx=40, marker_color="black", opacity=0.6, name="Polyline"), row=2, col=2)
    fig.update_layout(barmode="overlay", height=900, width=1200, title_text=f"Polyline diagnostics  {src_stage}→[{held_stage}]→{tgt_stage}")
    fig.update_xaxes(title_text="nodes", row=1, col=2)
    fig.update_xaxes(title_text="ratio", row=2, col=1)
    fig.update_xaxes(title_text="||v_t||", row=2, col=2)

    out = "polyline_diagnostic.html"
    fig.write_html(out)
    print(f"\n  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STARK et al. TOY EXPERIMENT (Fisher Flow Figure 3)
# ─────────────────────────────────────────────────────────────────────────────


def generate_stark_probs(K, n_components=4, seed=42):
    rng = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.rand(n_components, K, generator=rng), dim=-1)


def sample_stark_target(probs, n):
    k, K = probs.shape
    idx = torch.multinomial(probs.expand(n, -1, -1).reshape(-1, K), 1,
                            replacement=True).reshape(n, k)
    return torch.nn.functional.one_hot(idx, K).float()  # (n, k, K)


def sample_stark_source(n, k, K, device="cpu"):
    return torch.distributions.Dirichlet(
        torch.ones(k, K, device=device)
    ).sample((n,))  # (n, k, K)


def smooth_stark_labels(one_hot, mx=0.9999):
    K = one_hot.shape[-1]
    eps = (1.0 - mx) / (K - 1)
    out = torch.full_like(one_hot, eps)
    out[one_hot == 1.0] = mx
    return out


def to_orthant_product(x_simplex):
    return torch.sqrt(x_simplex.clamp(min=1e-12))


def from_orthant_product(y_sphere):
    return y_sphere ** 2


def normalize_product_sphere(y, k, K):
    B = y.shape[0]
    y3 = y.view(B, k, K)
    y3 = y3 / y3.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return y3.view(B, k * K)


def project_product_tangent(v, z, k, K):
    B = v.shape[0]
    v3 = v.view(B, k, K)
    z3 = z.view(B, k, K)
    v3 = v3 - (v3 * z3).sum(dim=-1, keepdim=True) * z3
    return v3.view(B, k * K)


def compute_product_sphere_cost(Y0, Y1, k, K):
    """Squared product-geodesic cost on (S^{K-1}_+)^k. Stays on device until final .cpu()."""
    Y0_3 = Y0.view(-1, k, K)
    Y1_3 = Y1.view(-1, k, K)
    cost = torch.zeros(len(Y0_3), len(Y1_3), device=Y0.device)
    for c in range(k):
        cos_sim = (Y0_3[:, c] @ Y1_3[:, c].T).clamp(-1 + 1e-6, 1 - 1e-6)
        cost += torch.acos(cos_sim) ** 2
    return cost.cpu().numpy()


class ProductSLERPInterpolant:
    def __init__(self, Y0, Y1, k, K):
        self.Y0 = Y0
        self.Y1 = Y1
        self.k = k
        self.K = K
        self.device = Y0.device

    def sample(self, src_idx, tgt_idx, t):
        y0 = self.Y0[src_idx].view(-1, self.k, self.K)
        y1 = self.Y1[tgt_idx].view(-1, self.k, self.K)
        if t.dim() == 2:
            t = t.unsqueeze(1)  # (B, 1, 1)

        cos_omega = (y0 * y1).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)

        w0 = torch.sin((1 - t) * omega) / sin_omega
        w1 = torch.sin(t * omega) / sin_omega
        z_t = w0 * y0 + w1 * y1
        z_t = z_t / z_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        v_t = omega * (
            -torch.cos((1 - t) * omega) / sin_omega * y0
            + torch.cos(t * omega) / sin_omega * y1
        )
        v_t = v_t - (v_t * z_t).sum(dim=-1, keepdim=True) * z_t

        B = z_t.shape[0]
        return z_t.view(B, -1), v_t.view(B, -1)


class LinearSimplexInterpolant:
    def __init__(self, X0, X1, k, K):
        self.X0 = X0
        self.X1 = X1
        self.k = k
        self.K = K
        self.device = X0.device

    def sample(self, src_idx, tgt_idx, t):
        x0 = self.X0[src_idx]
        x1 = self.X1[tgt_idx]
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x_t = (1 - t) * x0 + t * x1
        v_t = x1 - x0
        return x_t, v_t


class StarkBlock(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim, resid=True):
        super().__init__()
        self.resid = resid and (in_dim == out_dim)
        self.net = nn.Linear(in_dim + emb_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, emb):
        out = self.net(torch.cat([x, emb], dim=-1))
        out = self.act(out)
        if self.resid:
            out = out + x
        return out


class StarkFlowNet(nn.Module):
    def __init__(self, k, K, hidden=512, depth=4, emb_size=64):
        super().__init__()
        self.k = k
        self.K = K
        fd = k * K
        self.time_emb = nn.Linear(1, emb_size)
        layers = []
        for i in range(depth):
            ind = fd if i == 0 else hidden
            out = hidden if i < depth - 1 else fd
            layers.append(StarkBlock(ind, out, emb_size, resid=(ind == out)))
        # Last block has no activation
        layers[-1].act = nn.Identity()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        emb = self.time_emb(t)
        for layer in self.layers:
            x = layer(x, emb)
        return x


def train_product_riemannian_score(
    cells, k, K,
    n_iters=3000, batch_size=256, lr=3e-4,
    sigma_min=0.02, sigma_max=1.0,
    lognormal_mean=-1.2, lognormal_std=1.2,
    n_brownian_steps=3,
    hidden=256, depth=4, label="ScoreNet",
):
    """
    Denoising score matching on the product of spheres (S^{K-1}_+)^k.
    Per-component tangent projection, Exp, and log maps.

    Uses the same improvements as train_riemannian_score:
      - Wider σ range [0.02, 1.0]
      - Lognormal σ sampling (EDM schedule)
      - Multi-step per-component Brownian perturbation
      - SigmaEmbedding (Fourier features) in RiemannianScoreNet
    """
    D = k * K
    model = RiemannianScoreNet(D, hidden=hidden, depth=depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = len(cells)
    cells_dev = cells.to(DEVICE)

    losses = []
    for i in range(n_iters):
        idx = np.random.randint(0, N, size=batch_size)
        c = cells_dev[idx]  # (B, k*K)
        c3 = c.view(batch_size, k, K)

        # Lognormal σ
        log_sigma_b = sample_log_sigma_lognormal(
            batch_size, DEVICE, sigma_min, sigma_max,
            log_mean=lognormal_mean, log_std=lognormal_std,
        )  # (B, 1)
        log_sigma = log_sigma_b.view(batch_size, 1, 1)  # (B, 1, 1) for broadcasting
        sigma = log_sigma.exp()  # (B, 1, 1)

        # Per-component multi-step Brownian
        z3 = product_sphere_brownian_perturb(c3, sigma, n_steps=n_brownian_steps)

        # Per-component log map: log_z(c)
        cos_om = (z3 * c3).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        om = torch.acos(cos_om)
        sin_om = torch.sin(om).clamp(min=1e-8)
        log_z_c = (om / sin_om) * (c3 - cos_om * z3)
        log_z_c = log_z_c - (log_z_c * z3).sum(-1, keepdim=True) * z3

        target = log_z_c / (sigma * sigma)  # (B, k, K)

        z_flat = z3.view(batch_size, -1)
        s_pred = model(z_flat, log_sigma_b.squeeze(1)).view(batch_size, k, K)

        loss = (((s_pred - target) * sigma) ** 2).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 200 == 0:
            losses.append(loss.item())
            print(f"  {label:22s} iter {i:4d}  loss={loss.item():.4f}")

    model.eval()
    return model, losses


def generate_product_fisher_flow(model, Y0, k, K, n_steps=100):
    model.eval()
    with torch.no_grad():
        zt = Y0.clone().to(DEVICE)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = torch.full((len(zt),), step * dt, device=DEVICE)
            v = model(zt, t_val)
            v = project_product_tangent(v, zt, k, K)
            zt = zt + dt * v
            zt = normalize_product_sphere(zt, k, K)
    return zt.cpu()


def generate_linear_flow(model, X0, k, K, n_steps=100):
    model.eval()
    B = len(X0)
    with torch.no_grad():
        xt = X0.clone().to(DEVICE)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t_val = torch.full((len(xt),), step * dt, device=DEVICE)
            v = model(xt, t_val)
            # Project to simplex tangent (sum to 0 per component)
            v3 = v.view(B, k, K)
            v3 = v3 - v3.mean(dim=-1, keepdim=True)
            v = v3.view(B, k * K)
            xt = xt + dt * v
            # Project back to simplex per component
            xt3 = xt.view(B, k, K).clamp(min=1e-12)
            xt3 = xt3 / xt3.sum(dim=-1, keepdim=True)
            xt = xt3.view(B, k * K)
    return xt.cpu()


@torch.inference_mode()
def evaluate_stark_kl(model, probs, k, K, is_linear=False,
                      n_eval=512_000, batch_size=2048, n_steps=100):
    acc = torch.zeros(k, K, dtype=torch.int64)
    total = 0
    for start in range(0, n_eval, batch_size):
        bs = min(batch_size, n_eval - start)
        x0_simplex = sample_stark_source(bs, k, K, device=DEVICE)  # (bs, k, K)
        if is_linear:
            x0_flat = x0_simplex.view(bs, -1)
            x1_flat = generate_linear_flow(model, x0_flat, k, K, n_steps)
            x1 = x1_flat.view(bs, k, K).cpu()
        else:
            y0 = to_orthant_product(x0_simplex).view(bs, -1)
            y0 = normalize_product_sphere(y0, k, K)
            y1 = generate_product_fisher_flow(model, y0, k, K, n_steps)
            x1 = from_orthant_product(y1.view(bs, k, K))
        categories = x1.argmax(dim=-1)  # (bs, k)
        one_hot = torch.nn.functional.one_hot(categories, K)
        acc += one_hot.sum(dim=0)
        total += bs
    emp = acc.float() / total
    probs_cpu = probs.cpu()
    kl = (emp * (emp.clamp(min=1e-12).log() - probs_cpu.log())).sum(dim=-1).mean()
    return kl.item()


def train_stark_flow(
    probs, k, K,
    use_ot=True, is_linear=False,
    score_net=None, alpha=0.0, score_net_sigma=0.1,
    n_iters=50000, batch_size=512, lr=1e-3,
    label="", n_train=100_000,
):
    D_flat = k * K
    model = StarkFlowNet(k, K).to(DEVICE)
    if USE_AMP:
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"  torch.compile failed ({e}); running uncompiled")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Pre-generate target pool on device
    target_oh = sample_stark_target(probs, n_train)  # (n_train, k, K)
    if is_linear:
        target_pool = target_oh.view(n_train, -1).to(DEVICE)
    else:
        target_pool = normalize_product_sphere(
            to_orthant_product(smooth_stark_labels(target_oh)).view(n_train, -1),
            k, K,
        ).to(DEVICE)

    # Pre-compute OT coupling from a K-scaled pool.
    #
    # Use EMD (exact OT) at low K where the pool is small enough to be
    # tractable — EMD produces a hard permutation which preserves sharp
    # coupling structure. Use Sinkhorn (regularized OT) only at high K
    # where EMD becomes too slow on a large pool. Sinkhorn gives a smoothed
    # plan which hurts K=20 performance by ~60% (FF(OT) KL 0.010 → 0.016
    # observed), so we only use it when EMD would be impractical.
    if use_ot and not is_linear:
        n_ot_pool = min(n_train, max(5000, K * 200))
        # EMD up to pool=15000 (covers K≤75, takes ~5 min on one core)
        # Sinkhorn only above that for tractability
        use_sinkhorn = n_ot_pool > 15000
        solver = "Sinkhorn" if use_sinkhorn else "EMD (exact)"
        print(f"  Pre-computing OT coupling (pool={n_ot_pool}, {solver})...")
        src_pool_simp = sample_stark_source(n_ot_pool, k, K, device=DEVICE)
        src_pool = normalize_product_sphere(
            to_orthant_product(src_pool_simp).view(n_ot_pool, -1), k, K,
        )
        tgt_sub = target_pool[:n_ot_pool]
        cost = compute_product_sphere_cost(src_pool, tgt_sub, k, K)
        a = np.ones(n_ot_pool) / n_ot_pool
        b = np.ones(n_ot_pool) / n_ot_pool
        if HAS_POT:
            if use_sinkhorn:
                # Stabilized Sinkhorn at small ε: near-EMD sharpness
                reg = 0.005 * cost.max()
                T = ot.sinkhorn(
                    a, b, cost, reg=reg,
                    method="sinkhorn_stabilized",
                    numItermax=2000, stopThr=1e-7,
                )
            else:
                T = ot.emd(a, b, cost)
            T_flat = T.flatten()
            T_flat = np.maximum(T_flat, 0)
            T_flat = T_flat / T_flat.sum()
            n_ot_pairs = min(500_000, n_ot_pool * 10)
            flat_idx = np.random.choice(len(T_flat), size=n_ot_pairs, p=T_flat)
            ot_src_idx = flat_idx // n_ot_pool
            ot_tgt_idx = flat_idx % n_ot_pool
            ot_src_pool = src_pool[ot_src_idx]
            ot_tgt_pool = tgt_sub[ot_tgt_idx]
            print(f"  OT coupling: {n_ot_pairs} sampled pairs")
        else:
            ot_src_pool = src_pool
            ot_tgt_pool = tgt_sub
            n_ot_pairs = n_ot_pool

    use_score = alpha > 0.0 and score_net is not None
    if use_score:
        score_net.eval()
        log_sig = torch.full((batch_size,), float(np.log(score_net_sigma)), device=DEVICE)
        print(f"  Score reg: alpha={alpha}, sigma={score_net_sigma:.4f}")

    losses = []
    for it in range(n_iters):
        if use_ot and not is_linear:
            # Sample from pre-computed OT coupling
            idx = torch.randint(0, n_ot_pairs, (batch_size,), device=DEVICE)
            x0 = ot_src_pool[idx]
            x1 = ot_tgt_pool[idx]
        else:
            idx1 = np.random.randint(0, n_train, size=batch_size)
            x1 = target_pool[idx1]
            if is_linear:
                x0 = sample_stark_source(batch_size, k, K, device=DEVICE).view(batch_size, -1)
            else:
                x0_simp = sample_stark_source(batch_size, k, K, device=DEVICE)
                x0 = normalize_product_sphere(
                    to_orthant_product(x0_simp).view(batch_size, -1), k, K,
                )

        t = torch.rand(batch_size, 1, device=DEVICE)

        if is_linear:
            interp_x = (1 - t) * x0 + t * x1
            interp_v = x1 - x0
        else:
            y0_3 = x0.view(batch_size, k, K)
            y1_3 = x1.view(batch_size, k, K)
            t3 = t.unsqueeze(-1)  # (B, 1, 1)
            cos_om = (y0_3 * y1_3).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
            om = torch.acos(cos_om)
            sin_om = torch.sin(om).clamp(min=1e-8)
            w0 = torch.sin((1 - t3) * om) / sin_om
            w1 = torch.sin(t3 * om) / sin_om
            z3 = w0 * y0_3 + w1 * y1_3
            z3 = z3 / z3.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v3 = om * (-torch.cos((1 - t3) * om) / sin_om * y0_3
                       + torch.cos(t3 * om) / sin_om * y1_3)
            v3 = v3 - (v3 * z3).sum(-1, keepdim=True) * z3
            interp_x = z3.view(batch_size, -1)
            interp_v = v3.view(batch_size, -1)

        if use_score:
            with torch.no_grad():
                score = score_net(interp_x, log_sig[:len(interp_x)])
            interp_v = interp_v + alpha * score

        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            v_pred = model(interp_x.detach(), t.squeeze(1))
            if is_linear:
                v3 = v_pred.view(batch_size, k, K)
                v3 = v3 - v3.mean(dim=-1, keepdim=True)
                v_pred = v3.view(batch_size, -1)
            else:
                v_pred = project_product_tangent(v_pred, interp_x.detach(), k, K)
            loss = ((v_pred - interp_v.detach()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if it % 2000 == 0:
            losses.append(loss.item())
            print(f"  {label:30s} iter {it:5d}  loss={loss.item():.6f}")

    return model, losses


def main_stark(K_values=None, n_iters=50000, n_seeds=1, score_alpha=0.005,
               score_net_sigma=0.7, score_iters=3000):
    if K_values is None:
        K_values = [20, 40, 60, 80, 100, 120, 140, 160]
    k = 4

    methods = {
        "FF (No OT)":       dict(use_ot=False, is_linear=False, alpha=0.0),
        "FF (OT)":          dict(use_ot=True,  is_linear=False, alpha=0.0),
        "FF (OT) + Score":  dict(use_ot=True,  is_linear=False, alpha=score_alpha),
        "Linear":           dict(use_ot=False, is_linear=True,  alpha=0.0),
    }

    results = {name: {K: [] for K in K_values} for name in methods}

    for K in K_values:
        for seed in range(n_seeds):
            probs = generate_stark_probs(K, k, seed)
            print(f"\n{'='*60}")
            print(f"K={K}, seed={seed}")
            print(f"{'='*60}")

            # Train score net for this K (shared across score methods)
            score_net = None
            need_score = any(m["alpha"] > 0 for m in methods.values())
            if need_score:
                score_n_iters = max(score_iters, score_iters * K // 20)
                print(f"\n  Training score net (K={K}, iters={score_n_iters})...")
                target_oh = sample_stark_target(probs, 50000)
                target_sphere = normalize_product_sphere(
                    to_orthant_product(smooth_stark_labels(target_oh)).view(-1, k * K),
                    k, K,
                ).to(DEVICE)
                # Align score training sampling with the query σ used at
                # inference (score_net_sigma=0.7). Default lognormal_mean=-1.2
                # peaks at σ≈0.3 which is a query/training distribution
                # mismatch — fix by centering on log(score_net_sigma).
                score_net, _ = train_product_riemannian_score(
                    target_sphere, k, K, n_iters=score_n_iters,
                    batch_size=256, sigma_min=0.05, sigma_max=1.0,
                    lognormal_mean=float(np.log(score_net_sigma)),
                    lognormal_std=0.6,
                    hidden=512,
                    label=f"ScoreNet(K={K})",
                )

            for name, cfg in methods.items():
                print(f"\n  Training {name} (K={K}, seed={seed})...")
                sn = score_net if cfg["alpha"] > 0 else None
                model, _ = train_stark_flow(
                    probs, k, K,
                    use_ot=cfg["use_ot"],
                    is_linear=cfg["is_linear"],
                    score_net=sn, alpha=cfg["alpha"],
                    score_net_sigma=score_net_sigma,
                    n_iters=n_iters, label=name,
                )
                print(f"  Evaluating {name} (512k samples)...")
                kl = evaluate_stark_kl(
                    model, probs.to(DEVICE), k, K,
                    is_linear=cfg["is_linear"],
                )
                results[name][K].append(kl)
                print(f"  {name}: KL = {kl:.6f}")

    # Print summary table
    print(f"\n{'='*60}")
    print("STARK EXPERIMENT RESULTS (KL divergence, lower is better)")
    print(f"{'='*60}")
    header = f"  {'Method':<22}" + "".join(f"  K={K:>3}" for K in K_values)
    print(header)
    print("  " + "-" * (22 + 7 * len(K_values)))
    for name in methods:
        row = "".join(f"  {np.mean(results[name][K]):>5.3f}" for K in K_values)
        print(f"  {name:<22}{row}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"FF (No OT)": "#1f77b4", "FF (OT)": "#ff7f0e",
              "FF (OT) + Score": "#2ca02c", "Linear": "#d62728"}
    for name in methods:
        means = [np.mean(results[name][K]) for K in K_values]
        stds = [np.std(results[name][K]) if len(results[name][K]) > 1 else 0
                for K in K_values]
        ax.errorbar(K_values, means, yerr=stds, label=name,
                    color=colors.get(name, "#888"), marker="o", capsize=3)
    ax.set_xlabel("K (categories per simplex)")
    ax.set_ylabel("KL divergence")
    ax.set_title("Stark et al. Toy Experiment")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("stark_experiment.png", dpi=150)
    print("\nSaved stark_experiment.png")
    plt.close()


def main_conflict_sweep(ot_subsample=2000, knn=15, anchor_values=(1, 2, 3, 5, 10)):
    """
    Pre-training diagnostic for the sparse-anchor sweep. Measures the
    waypoint-conflict rate — the % of interior cells visited by ≥2 OT pairs
    that carry ≥2 distinct outgoing directions — as a function of n_anchors.
    The full polyline has ~85% conflict (see polyline_diagnostic.html); the
    hypothesis for Sparse Anchor Flow Matching is that n_anchors=1..3 brings
    this to near 0 while still anchoring the interpolant to real cells.

    Runs on the first LOO transition only and prints a single table. Cheap
    (< 1 minute) — use it to validate the hypothesis before spending ~45 min
    on the full training sweep.
    """
    from collections import defaultdict

    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    data = load_embryoid_body("embryoid_body.h5ad", n_hvg=2000)
    stages = data["train"]["stages"]
    src_stage, held_stage, tgt_stage = stages[0], stages[1], stages[2]

    print(f"\n{'=' * 60}")
    print(f"CONFLICT SWEEP  {src_stage} -> [{held_stage}] -> {tgt_stage}")
    print("=" * 60)

    P_src = to_compositional(data["train"]["cells"][src_stage])
    P_tgt = to_compositional(data["train"]["cells"][tgt_stage])
    Y0 = normalize_sphere(to_orthant(P_src)).to(DEVICE)
    Y1 = normalize_sphere(to_orthant(P_tgt)).to(DEVICE)
    Y0_sub, _ = _subsample_tensor(Y0, ot_subsample, rng)
    Y1_sub, _ = _subsample_tensor(Y1, ot_subsample, rng)

    print(f"\n  Building polyline (n={len(Y0_sub)}+{len(Y1_sub)}, k={knn})...")
    base = PolylineGeodesic(Y0_sub, Y1_sub, k=knn)
    cost_graph = base.cost_matrix

    n_pool = 5000
    print(f"  Sampling {n_pool} OT pairs via graph cost...")
    ot_src, ot_tgt = ot_coupling(cost_graph, n_pool)

    def measure(interp, label):
        paths = [interp.reconstruct_path(int(s), int(g)) for s, g in zip(ot_src, ot_tgt)]
        lens = np.array([len(p) for p in paths])
        next_map = defaultdict(lambda: defaultdict(int))
        for p in paths:
            for i in range(len(p) - 1):
                next_map[p[i]][p[i + 1]] += 1
        reused = [nxt for nxt in next_map.values() if sum(nxt.values()) >= 2]
        distinct = np.array([len(nxt) for nxt in reused]) if reused else np.array([])
        conflict = (distinct > 1).mean() * 100 if len(distinct) else 0.0
        concentration = np.array([
            sum((c / sum(nxt.values())) ** 2 for c in nxt.values()) for nxt in reused
        ]).mean() if reused else 1.0
        print(f"  {label:<20} mean_len={lens.mean():5.2f}  reused_nodes={len(reused):5d}  "
              f"conflict={conflict:5.1f}%  concentration={concentration:.3f}")

    print(f"\n  {'Method':<20} {'path nodes':>10}  {'reused':>12}  {'conflict':>10}  concentration")
    print("  " + "-" * 76)
    measure(base, "Full polyline")
    for n in anchor_values:
        interp = SparseAnchorGeodesic(base, n_anchors=n)
        measure(interp, f"Anchor{n}")

    print(f"\n  concentration: 1.0 = every reused node carries one direction (single-valued field)")
    print(f"                 < 1.0 = multiple outgoing directions (multi-valued, hard for the net to fit)")
    print(f"\n  If Anchor1..3 show conflict rate < 10%, the hypothesis is validated —")
    print(f"  proceed with:  uv run python main.py embryoid --methods Sphere+SLERP,Graph+Polyline,Graph+Anchor1,Graph+Anchor2,Graph+Anchor3,Graph+Anchor5")


MM_METHOD_NAMES = [
    "MM+SLERP", "MM+Score_kde", "MM+Score_learned", "MM+Score_forward",
    "MM+SLERP+PHATE", "MM+SLERP+PHATE-Euclidean",
    "MM+SLERP+PHATE-SphereArc",
    "MM+SLERP+SquaredSpectral",
    "MM+SLERP+Biharmonic", "MM+SLERP+GlobalBiharmonic",
    "MM+Score_learned+PHATE", "MM+Score_learned+Biharmonic",
    "MM+Score_timed", "MM+Score_timed+Biharmonic",
    "MM+SI", "MM+SI+Biharmonic",
    "MM+BiharmonicVel", "MM+BiharmonicWaypoint",
]


def make_true_biharmonic_cost_fn():
    """Legacy CLI helper for a cost that always means true biharmonic."""

    def biharmonic_cost(Y0, Y1):
        return compute_biharmonic_cost_matrix(Y0, Y1, weight_power=1.0)

    biharmonic_cost.__name__ = "biharmonic_cost"
    return biharmonic_cost


def main_mm(n_iters=3000, batch_size=256, ot_subsample=2000,
            score_iters=3000, score_alpha=0.1, score_net_sigma=0.1,
            methods=None, n_seeds=1, holdout_stages=None, otpfm_holdout=False,
            si_sigma=0.05, inf_sigma=0.0, biharm_beta=0.3, visualize=False):
    """
    Multi-marginal Fisher Flow Matching with OTP-FM eval protocol.

    Trains a single velocity field jointly across all 5 stages as marginals
    at times [0, 0.25, 0.5, 0.75, 1.0]. Evaluates by integrating from held-out
    test cells at t=0 to each stage's time and comparing to held-out test
    cells at that stage (MMD² in compositional space).

    When n_seeds>1, repeats the entire train+eval pipeline for each seed and
    reports mean ± std across seeds. Seeds affect score net init, flow model
    init, OT subsampling, and minibatch ordering.
    """
    data = load_embryoid_body("embryoid_body.h5ad", n_hvg=2000)
    stages = data["train"]["stages"]
    S = len(stages)
    stage_times = [i / (S - 1) for i in range(S)]  # [0, 0.25, 0.5, 0.75, 1.0]
    D = data["train"]["cells"][stages[0]].shape[1]

    print(f"\n  Stages: {stages}")
    print(f"  Times:  {[f'{t:.2f}' for t in stage_times]}")
    print(f"  Gene dim: {D}")

    # Train stage cells on device, mapped to the positive orthant
    train_stage_cells = [
        normalize_sphere(to_orthant(to_compositional(data["train"]["cells"][s]))).to(DEVICE)
        for s in stages
    ]
    test_stage_comp = [to_compositional(data["test"]["cells"][s]) for s in stages]
    test_stage_sphere = [normalize_sphere(to_orthant(tc)).to(DEVICE) for tc in test_stage_comp]

    # Resolve holdout pattern. --mm-holdout-stages takes precedence over
    # --mm-otpfm-holdout (which is a shortcut for "1,3" on S=5 data).
    if holdout_stages is not None:
        try:
            held_list = [int(x) for x in holdout_stages.split(",") if x.strip() != ""]
        except ValueError:
            raise SystemExit(f"--mm-holdout-stages must be comma-separated ints; got {holdout_stages!r}")
        for i in held_list:
            if not (0 <= i < S):
                raise SystemExit(f"--mm-holdout-stages index {i} out of range [0, {S-1}]")
        if 0 in held_list or (S - 1) in held_list:
            raise SystemExit("Cannot hold out the first or last stage (no valid training interval).")
        held_set = set(held_list)
    elif otpfm_holdout:
        if S < 3:
            raise SystemExit(f"--mm-otpfm-holdout needs S>=3 stages; got S={S}")
        held_set = {i for i in range(S) if i % 2 == 1}
    else:
        held_set = set()

    train_idx = [i for i in range(S) if i not in held_set]
    if held_set:
        print(f"\n  [holdout] training on stages {[stages[i] for i in train_idx]}"
              f" (times {[f'{stage_times[i]:.2f}' for i in train_idx]})")
        print(f"  [holdout] held-out stages:  {[stages[i] for i in sorted(held_set)]}"
              f" (times {[f'{stage_times[i]:.2f}' for i in sorted(held_set)]})")

    train_stage_cells_sub = [train_stage_cells[i] for i in train_idx]
    train_stage_times_sub = [stage_times[i] for i in train_idx]
    train_idx_set = set(train_idx)

    # Column labels for result tables (chained eval columns = stages 1..S-1).
    # When a holdout is active, tag each column as TRAIN (simulation fidelity)
    # or HOLDOUT (interpolation) and compute split means.
    chained_col_labels = [f"t={stage_times[i]:.2f}" for i in range(1, S)]
    perseg_col_labels = [f"t={stage_times[i]:.2f}→{stage_times[i+1]:.2f}" for i in range(S - 1)]
    if held_set:
        chained_holdout_cols = [j for j, i in enumerate(range(1, S)) if i not in train_idx_set]
        chained_train_cols = [j for j, i in enumerate(range(1, S)) if i in train_idx_set]
        # A per-segment hop (i → i+1) is "training" only if BOTH endpoints
        # were in training (the model saw this exact pair as adjacent).
        perseg_holdout_cols = [
            j for j, i in enumerate(range(S - 1))
            if not (i in train_idx_set and (i + 1) in train_idx_set)
        ]
        perseg_train_cols = [
            j for j, i in enumerate(range(S - 1))
            if i in train_idx_set and (i + 1) in train_idx_set
        ]
    else:
        chained_holdout_cols = chained_train_cols = []
        perseg_holdout_cols = perseg_train_cols = []

    # Method filter
    method_list = list(MM_METHOD_NAMES)
    if methods is not None:
        requested = [m.strip() for m in methods.split(",")]
        unknown = [m for m in requested if m not in MM_METHOD_NAMES]
        if unknown:
            raise SystemExit(f"Unknown MM methods: {unknown}. Choices: {MM_METHOD_NAMES}")
        method_list = [m for m in MM_METHOD_NAMES if m in requested]
        print(f"  [filter] running methods: {method_list}")

    print(f"  n_seeds: {n_seeds}")

    need_learned = any(
        m in method_list
        for m in ("MM+Score_learned", "MM+Score_learned+PHATE", "MM+Score_learned+Biharmonic")
    )
    need_forward = "MM+Score_forward" in method_list
    need_kde = "MM+Score_kde" in method_list
    need_timed = any(
        m in method_list
        for m in ("MM+Score_timed", "MM+Score_timed+Biharmonic")
    )
    need_global_biharmonic = "MM+SLERP+GlobalBiharmonic" in method_list
    need_biharm_vel = "MM+BiharmonicVel" in method_list
    need_biharm_wp = "MM+BiharmonicWaypoint" in method_list
    need_any_global_biharm = need_global_biharmonic or need_biharm_vel or need_biharm_wp

    # Precompute global biharmonic embedding (deterministic, shared across seeds)
    global_biharmonic_cost_fn = None
    global_biharm_embeddings = None
    if need_any_global_biharm:
        print("  Computing global biharmonic embedding across all training stages...")
        global_biharm_embeddings = compute_global_biharmonic_embedding(train_stage_cells_sub)
        if need_global_biharmonic:
            global_biharmonic_cost_fn = make_global_biharmonic_cost_fn(train_stage_cells_sub)

    # ── Per-seed accumulators ────────────────────────────────────────────
    # chained_by_method[name] = list of (S-1)-length row arrays, one per seed
    chained_by_method = {}
    perseg_by_method = {}
    infer_chained = {a: [] for a in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)}
    infer_perseg = {a: [] for a in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)}

    # ── Per-seed train + eval loop ───────────────────────────────────────
    for seed_idx in range(n_seeds):
        seed = 42 + seed_idx
        print("\n" + "#" * 60)
        print(f"# SEED {seed_idx + 1}/{n_seeds}  (base={seed})")
        print("#" * 60)
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # ── Train the global Riemannian score network (all stages) ──────
        score_net = None
        if need_learned:
            print("\n" + "=" * 60)
            print("TRAINING RIEMANNIAN SCORE NETWORK (DSM, global)")
            print("=" * 60)
            all_training = torch.cat(train_stage_cells_sub, dim=0)
            print(f"  Training on {len(all_training)} cells (union of {len(train_stage_cells_sub)} stages)")
            score_net, _ = train_riemannian_score(
                all_training, D, n_iters=score_iters, batch_size=batch_size,
            )

        # ── Train per-interval forward-directed score nets ──────────────
        forward_score_nets = None
        if need_forward:
            print("\n" + "=" * 60)
            print("TRAINING FORWARD-DIRECTED SCORE NETS (one per interval)")
            print("=" * 60)
            forward_score_nets = train_forward_score_nets(
                train_stage_cells_sub, D, n_iters=score_iters, batch_size=batch_size,
            )

        # ── Train the time-conditioned score network ────────────────────
        timed_score_net = None
        if need_timed:
            print("\n" + "=" * 60)
            print("TRAINING TIME-CONDITIONED SCORE NETWORK (DSM)")
            print("=" * 60)
            print(f"  Training on {sum(len(c) for c in train_stage_cells_sub)} cells"
                  f" across {len(train_stage_cells_sub)} stages (per-stage conditional)")
            timed_score_net, _ = train_timed_riemannian_score(
                train_stage_cells_sub, train_stage_times_sub, D,
                n_iters=score_iters, batch_size=batch_size,
            )

        # ── Precompute KDE score cloud (for the KDE baseline method) ────
        kde_cells = None
        kde_sigma = None
        if need_kde:
            print("\n  Building KDE score cloud (for MM+Score_kde baseline)...")
            all_training = torch.cat(train_stage_cells_sub, dim=0)
            kde_cells, _ = _subsample_tensor(all_training, 2000, rng)
            kde_sigma = estimate_rbf_sigma(all_training, k=50)
            print(f"    {len(kde_cells)} cells, sigma={kde_sigma:.4f}")

        # ── Train each multi-marginal method ────────────────────────────
        models = {}
        mm_losses = {}
        for name in method_list:
            print("\n" + "=" * 60)
            print(f"TRAINING {name}  (seed {seed_idx + 1}/{n_seeds})")
            print("=" * 60)
            kwargs = {}
            if name == "MM+Score_kde":
                kwargs = {
                    "score_net": _KDEScoreWrapper(kde_cells, kde_sigma),
                    "alpha": score_alpha,
                    "score_net_sigma": kde_sigma,
                }
            elif name == "MM+Score_learned":
                kwargs = {
                    "score_net": score_net,
                    "alpha": score_alpha,
                    "score_net_sigma": score_net_sigma,
                }
            elif name == "MM+Score_forward":
                kwargs = {
                    "score_nets_per_interval": forward_score_nets,
                    "alpha": score_alpha,
                    "score_net_sigma": score_net_sigma,
                }
            elif name == "MM+SLERP+PHATE":
                kwargs = {"cost_fn": compute_phate_cost_matrix}
            elif name == "MM+SLERP+PHATE-Euclidean":
                kwargs = {"cost_fn": make_phate_cost_fn(graph_metric="euclidean")}
            elif name == "MM+SLERP+PHATE-SphereArc":
                kwargs = {"cost_fn": make_phate_cost_fn(graph_metric="sphere_arc")}
            elif name == "MM+SLERP+SquaredSpectral":
                kwargs = {"cost_fn": compute_biharmonic_cost_matrix}
            elif name == "MM+SLERP+Biharmonic":
                kwargs = {"cost_fn": make_true_biharmonic_cost_fn()}
            elif name == "MM+SLERP+GlobalBiharmonic":
                kwargs = {"cost_fn": global_biharmonic_cost_fn}
            elif name == "MM+Score_learned+PHATE":
                kwargs = {
                    "score_net": score_net,
                    "alpha": score_alpha,
                    "score_net_sigma": score_net_sigma,
                    "cost_fn": compute_phate_cost_matrix,
                }
            elif name == "MM+Score_learned+Biharmonic":
                kwargs = {
                    "score_net": score_net,
                    "alpha": score_alpha,
                    "score_net_sigma": score_net_sigma,
                    "cost_fn": compute_biharmonic_cost_matrix,
                }
            elif name == "MM+Score_timed":
                kwargs = {
                    "score_net": timed_score_net,
                    "alpha": score_alpha,
                    "score_net_sigma": score_net_sigma,
                }
            elif name == "MM+Score_timed+Biharmonic":
                kwargs = {
                    "score_net": timed_score_net,
                    "alpha": score_alpha,
                    "score_net_sigma": score_net_sigma,
                    "cost_fn": compute_biharmonic_cost_matrix,
                }
            elif name == "MM+SI":
                kwargs = {"si_sigma": si_sigma}
            elif name == "MM+SI+Biharmonic":
                kwargs = {
                    "si_sigma": si_sigma,
                    "cost_fn": compute_biharmonic_cost_matrix,
                }
            elif name == "MM+BiharmonicVel":
                kwargs = {
                    "biharm_beta": biharm_beta,
                    "global_biharm_embeddings": global_biharm_embeddings,
                }
            elif name == "MM+BiharmonicWaypoint":
                kwargs = {
                    "biharm_waypoints": True,
                    "global_biharm_embeddings": global_biharm_embeddings,
                }
            models[name], mm_losses[name] = train_multi_marginal_flow(
                train_stage_cells_sub, train_stage_times_sub, D,
                n_iters=n_iters, batch_size=batch_size, label=name,
                ot_subsample=ot_subsample, **kwargs,
            )

        # ── Stage-width reference (first seed only) ─────────────────────
        if seed_idx == 0:
            print("\n  Stage-width reference (train vs test MMD² per stage — generative floor):")
            for i, s in enumerate(stages):
                train_comp_s = to_compositional(data["train"]["cells"][s])
                test_comp_s = test_stage_comp[i]
                idx = rng.choice(len(train_comp_s), size=min(len(train_comp_s), len(test_comp_s)), replace=False)
                floor = mmd_rbf(train_comp_s[idx], test_comp_s)
                print(f"    {s} (t={stage_times[i]:.2f}): {floor:.4f}")
            print("\n  Do-nothing baseline (stage-0 test cells vs target stage test cells):")
            source_test = test_stage_comp[0]
            for i in range(1, S):
                mmd = mmd_rbf(source_test, test_stage_comp[i])
                print(f"    {stages[i]} (t={stage_times[i]:.2f}): {mmd:.4f}")

        # ── Trajectory visualization (first seed only) ──────────────────
        if seed_idx == 0 and visualize:
            test_stage_log1p = [data["test"]["cells"][s] for s in stages]
            visualize_endpoints_phate(
                models, test_stage_log1p, test_stage_sphere,
                stage_times, stages,
            )
            visualize_nn_distance_histograms(
                models, test_stage_log1p, test_stage_sphere,
                stage_times, stages,
            )
            visualize_trajectories_phate(
                models, test_stage_log1p, test_stage_sphere,
                stage_times, stages,
            )

        # ── Per-seed eval ────────────────────────────────────────────────
        infer_alphas = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
        slerp_model = models.get("MM+SLERP")

        def eval_chained(model, score_n=None, a=0.0):
            source = test_stage_sphere[0]
            row = []
            for i in range(1, S):
                t_target = stage_times[i]
                n_steps = max(10, int(50 * t_target))
                pred_sphere = generate_fisher_flow(
                    model, source, n_steps=n_steps, t_start=0.0, t_end=t_target,
                    score_net=score_n, alpha=a, score_net_sigma=score_net_sigma,
                    inf_sigma=inf_sigma,
                )
                pred_comp = from_orthant(pred_sphere)
                row.append(mmd_rbf(pred_comp, test_stage_comp[i]))
            return np.array(row)

        def eval_per_segment(model, score_n=None, a=0.0):
            row = []
            for i in range(S - 1):
                src_sphere = test_stage_sphere[i]
                pred_sphere = generate_fisher_flow(
                    model, src_sphere, n_steps=50,
                    t_start=stage_times[i], t_end=stage_times[i + 1],
                    score_net=score_n, alpha=a, score_net_sigma=score_net_sigma,
                    inf_sigma=inf_sigma,
                )
                pred_comp = from_orthant(pred_sphere)
                row.append(mmd_rbf(pred_comp, test_stage_comp[i + 1]))
            return np.array(row)

        # Accumulate chained + per-segment for each method
        for mname, mmodel in models.items():
            chained_by_method.setdefault(mname, []).append(eval_chained(mmodel))
            perseg_by_method.setdefault(mname, []).append(eval_per_segment(mmodel))

        # Inference-time score sweep on MM+SLERP (only if score_net is available)
        if slerp_model is not None and score_net is not None:
            for a in infer_alphas:
                if a == 0.0:
                    continue
                infer_chained[a].append(eval_chained(slerp_model, score_net, a))
                infer_perseg[a].append(eval_per_segment(slerp_model, score_net, a))

    # ── Aggregate + print ────────────────────────────────────────────────
    def fmt(mean, std):
        return f"{mean:.4f}±{std:.4f}" if n_seeds > 1 else f"{mean:.4f}"

    def print_table(title, method_rows, infer_rows, col_names, holdout_cols, train_cols):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        col_w = 16 if n_seeds > 1 else 12

        # Tag each column with TRAIN/HOLD when any holdout is active
        if held_set and holdout_cols and train_cols:
            tagged_names = []
            for j, cn in enumerate(col_names):
                tag = "TR" if j in train_cols else "HO"
                tagged_names.append(f"{cn}[{tag}]")
            header_cols = tagged_names
            show_split = True
        else:
            header_cols = col_names
            show_split = False

        extra = ["hold_mean", "train_mean"] if show_split else []
        header = "  " + f"{'Method':<32}" + "  ".join(
            f"{s:>{col_w}}" for s in header_cols
        ) + f"  {'mean':>{col_w}}" + "".join(f"  {e:>{col_w}}" for e in extra)
        print(header)
        print("  " + "-" * (32 + (col_w + 2) * (len(header_cols) + 1 + len(extra))))

        def row_str(rows):
            arr = np.stack(rows)  # (n_seeds, n_cols)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            total = arr.mean(axis=1)  # per-seed mean across hops
            cols = "  ".join(f"{fmt(mean[i], std[i]):>{col_w}}" for i in range(len(mean)))
            total_str = f"{fmt(total.mean(), total.std()):>{col_w}}"
            s = cols + "  " + total_str
            if show_split:
                hold_per_seed = arr[:, holdout_cols].mean(axis=1)
                train_per_seed = arr[:, train_cols].mean(axis=1)
                s += "  " + f"{fmt(hold_per_seed.mean(), hold_per_seed.std()):>{col_w}}"
                s += "  " + f"{fmt(train_per_seed.mean(), train_per_seed.std()):>{col_w}}"
            return s

        for mname in method_list:
            if mname not in method_rows:
                continue
            print(f"  {mname:<32}{row_str(method_rows[mname])}")
        for a in sorted(infer_rows):
            if not infer_rows[a] or a == 0.0:
                continue
            label = f"MM+SLERP+infer_α={a}"
            print(f"  {label:<32}{row_str(infer_rows[a])}")

    print_table(
        f"CHAINED EVAL ({n_seeds} seed{'s' if n_seeds > 1 else ''})",
        chained_by_method, infer_chained,
        chained_col_labels, chained_holdout_cols, chained_train_cols,
    )
    print_table(
        f"PER-SEGMENT EVAL ({n_seeds} seed{'s' if n_seeds > 1 else ''})",
        perseg_by_method, infer_perseg,
        perseg_col_labels, perseg_holdout_cols, perseg_train_cols,
    )
    print("\n  (all values are MMD² in compositional space; lower is better)")
    if n_seeds > 1:
        print(f"  Format: mean±std over {n_seeds} seeds")


class _KDEScoreWrapper:
    """
    Adapter exposing the same (z, log_sigma) -> score signature as
    RiemannianScoreNet, backed by the non-parametric RBF KDE. Lets the
    multi-marginal trainer switch between KDE and learned score via a single
    `score_net` argument.
    """

    def __init__(self, cells, sigma):
        self.cells = cells
        self.sigma = sigma

    def eval(self):
        return self

    def __call__(self, z, log_sigma):
        return rbf_spherical_score(z, self.cells, self.sigma)


def main_embryoid(max_cells_per_stage=None, ot_subsample=2000, n_iters=3000,
                  batch_size=256, knn=15, methods=None, score_ncells=2000,
                  score_sigma=None):
    """
    Embryoid body experiment, data-faithful Fisher FM with 2x2 ablation.

    Evaluation is leave-one-timepoint-out: for each middle stage we train
    a flow between the stages on either side and evaluate the predicted
    intermediate state at t=0.5 against the true held-out middle stage.
    This is the only setting where data-faithful interpolants can actually
    differ from SLERP — endpoint metrics can't distinguish them.

    methods: optional comma-separated subset of METHOD_NAMES to run. Useful
    for restricting the grid when re-running specific ablations.

    score_ncells: size of the cell cloud used to evaluate the RBF-KDE
    Riemannian score for Sphere+Score methods. Drawn from the union of all
    training stages so the score captures the entire ribbon geometry, not
    just the endpoints of each LOO transition.

    score_sigma: override for the RBF bandwidth. When None, set automatically
    to the median 50-NN arc distance among the score cells.
    """
    global METHOD_NAMES
    if methods is not None:
        requested = [m.strip() for m in methods.split(",")]
        unknown = [m for m in requested if m not in METHOD_NAMES]
        if unknown:
            raise SystemExit(f"Unknown methods: {unknown}. Choices: {METHOD_NAMES}")
        METHOD_NAMES = [m for m in METHOD_NAMES if m in requested]
        print(f"  [filter] running methods: {METHOD_NAMES}")

    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    data = load_embryoid_body("embryoid_body.h5ad", n_hvg=2000, max_cells_per_stage=max_cells_per_stage)
    stages = data["train"]["stages"]
    D = data["train"]["cells"][stages[0]].shape[1]

    # LOO transitions: skip every middle stage
    loo_transitions = [
        (stages[i - 1], stages[i], stages[i + 1])
        for i in range(1, len(stages) - 1)
    ]
    print(f"\n  LOO transitions (src -> held -> tgt): {loo_transitions}")

    # ── Score cloud for RBF-KDE regularizer ──────────────────────────────
    # Built once per run from the union of all training stages, so the score
    # represents the full developmental ribbon. Only used by Sphere+Score*.
    any_score_method = any(m.startswith("Sphere+Score") for m in METHOD_NAMES)
    score_cells = None
    sigma = None
    if any_score_method:
        print("\n  Building score cloud for RBF-KDE regularizer...")
        all_stage_tensors = [
            normalize_sphere(to_orthant(to_compositional(data["train"]["cells"][s])))
            for s in stages
        ]
        full_cloud = torch.cat(all_stage_tensors, dim=0).to(DEVICE)
        # Sigma is estimated from the FULL cloud (all ~25k training cells) so
        # the local scale reflects the real data density. The subsample is
        # only used for the per-batch score evaluation to keep it cheap. This
        # matters a lot — estimating sigma from the subsample inflates it by
        # ~3× because sparsification pushes k-NN distances outward.
        if score_sigma is not None:
            sigma = float(score_sigma)
        else:
            sigma = estimate_rbf_sigma(full_cloud, k=50)
        score_cells, _ = _subsample_tensor(full_cloud, score_ncells, rng)
        print(f"    {len(score_cells)}/{len(full_cloud)} cells from {len(stages)} stages, "
              f"sigma={sigma:.4f} ({'manual' if score_sigma is not None else 'auto full-cloud median 50-NN arc'})")

    loo_results = {}
    loo_losses = {}

    for src_stage, held_stage, tgt_stage in loo_transitions:
        print("\n" + "=" * 60)
        print(f"LOO TRANSITION: {src_stage} -> [{held_stage}] -> {tgt_stage}")
        print("=" * 60)

        # Raw expression data
        P_src_train = to_compositional(data["train"]["cells"][src_stage])
        P_tgt_train = to_compositional(data["train"]["cells"][tgt_stage])
        P_src_test = to_compositional(data["test"]["cells"][src_stage])
        P_held_test = to_compositional(data["test"]["cells"][held_stage])
        P_tgt_test = to_compositional(data["test"]["cells"][tgt_stage])

        Y0_train = normalize_sphere(to_orthant(P_src_train)).to(DEVICE)
        Y1_train = normalize_sphere(to_orthant(P_tgt_train)).to(DEVICE)
        Y0_test = normalize_sphere(to_orthant(P_src_test)).to(DEVICE)

        print(f"  Train: {len(Y0_train)} source, {len(Y1_train)} target cells")
        print(f"  Test:  src={len(P_src_test)}, held={len(P_held_test)}, tgt={len(P_tgt_test)}")
        print(f"  Gene dim: {D}")

        # Subsample for OT / graph construction (quadratic cost)
        Y0_sub, _ = _subsample_tensor(Y0_train, ot_subsample, rng)
        Y1_sub, _ = _subsample_tensor(Y1_train, ot_subsample, rng)
        print(f"\n  Building ablation grid (subsample {len(Y0_sub)} x {len(Y1_sub)}, k={knn})...")

        ablation, cost_sphere, cost_graph, poly_interp = build_ablation(
            Y0_sub, Y1_sub, k=knn, score_cells=score_cells, sigma=sigma,
        )
        print(f"    Sphere cost range: [{cost_sphere.min():.3f}, {cost_sphere.max():.3f}]")
        print(f"    Graph  cost range: [{cost_graph.min():.3f}, {cost_graph.max():.3f}]")

        # Save OT coupling visualization for the first transition only
        if len(loo_results) == 0:
            plot_ot_coupling_comparison(
                cost_sphere, cost_graph, Y0_sub, Y1_sub,
                filename=f"ot_coupling_{src_stage}_{tgt_stage}.png",
            )

        # Train all four methods
        models, losses = train_ablation(ablation, D, n_iters, batch_size)
        loo_losses[(src_stage, held_stage, tgt_stage)] = losses

        # Generate predictions at t=0.5 (intermediate) and t=1.0 (endpoint)
        print("\n  Generating predictions (intermediate and endpoint)...")
        preds_intermediate = {}
        preds_endpoint = {}
        for name, model in models.items():
            pred_mid_y = generate_fisher_flow(model, Y0_test, n_steps=25, t_start=0.0, t_end=0.5)
            pred_end_y = generate_fisher_flow(model, Y0_test, n_steps=50, t_start=0.0, t_end=1.0)
            preds_intermediate[name] = from_orthant(pred_mid_y)
            preds_endpoint[name] = from_orthant(pred_end_y)

        loo_results[(src_stage, held_stage, tgt_stage)] = {
            "P_src_test": P_src_test,
            "P_held_test": P_held_test,
            "P_tgt_test": P_tgt_test,
            "preds_intermediate": preds_intermediate,
            "preds_endpoint": preds_endpoint,
        }

    # ── Aggregate metrics ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT METRICS")
    print("=" * 60)
    print(f"  {'Transition':<18} {'Method':<18} {'MMD²(t=0.5, held)':>20} {'MMD²(t=1, tgt)':>18}")
    print("  " + "-" * 76)
    for key, res in loo_results.items():
        src, held, tgt = key
        trans_label = f"{src}→[{held}]→{tgt}"
        for name in METHOD_NAMES:
            mmd_mid = mmd_rbf(res["preds_intermediate"][name], res["P_held_test"])
            mmd_end = mmd_rbf(res["preds_endpoint"][name], res["P_tgt_test"])
            print(f"  {trans_label:<18} {name:<18} {mmd_mid:>20.4f} {mmd_end:>18.4f}")
        print()

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_embryoid_loo_results(data, loo_results, stages)
    plot_embryoid_loss_curves(loo_losses)

    print("\nDone. Saved: embryoid_results.html, embryoid_loss_curves.html")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data-faithful Fisher Flow Matching on the positive orthant")
    parser.add_argument("mode", nargs="?", default="toy",
                        choices=["toy", "embryoid", "diagnose", "conflict", "mm", "stark"],
                        help="'toy' (default), 'embryoid' (LOO training), "
                             "'diagnose' (polyline post-mortem), 'conflict' "
                             "(anchor-count conflict sweep, no training), "
                             "'mm' (multi-marginal Fisher Flow + OTP-FM eval), or "
                             "'stark' (Stark et al. density learning, Figure 3)")
    parser.add_argument("--n-cells", type=int, default=None,
                        help="Max cells per timepoint for embryoid (e.g. 500). Default: use all.")
    parser.add_argument("--ot-subsample", type=int, default=2000,
                        help="Max cells per side for OT cost + kNN graph (default: 2000). Quadratic cost.")
    parser.add_argument("--n-iters", type=int, default=3000,
                        help="Training iterations per (method, transition).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--knn", type=int, default=15,
                        help="Neighbors in the kNN graph underlying PolylineGeodesic and the graph OT cost.")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated subset of methods to train in embryoid mode. "
                             "Choices: " + ",".join(METHOD_NAMES))
    parser.add_argument("--score-ncells", type=int, default=2000,
                        help="Size of the RBF-KDE score cloud for Sphere+Score methods (default: 2000).")
    parser.add_argument("--score-sigma", type=float, default=None,
                        help="Override RBF bandwidth (arc-length units). Default: median 50-NN arc.")
    parser.add_argument("--mm-alpha", type=float, default=0.1,
                        help="Score regularization strength for MM methods (default: 0.1).")
    parser.add_argument("--mm-score-iters", type=int, default=3000,
                        help="Training iterations for the Riemannian score net in mm mode (default: 3000).")
    parser.add_argument("--mm-score-net-sigma", type=float, default=0.1,
                        help="Fixed noise level used when querying the learned score net during MM flow training (default: 0.1).")
    parser.add_argument("--mm-seeds", type=int, default=1,
                        help="Number of random seeds for MM mode (default: 1). Aggregates mean±std.")
    parser.add_argument("--mm-otpfm-holdout", action="store_true",
                        help="OTP-FM eval protocol: shortcut for --mm-holdout-stages 1,3 on the "
                             "5-stage embryoid body run (train on {t0,t2,t4}).")
    parser.add_argument("--mm-holdout-stages", type=str, default=None,
                        help="Comma-separated stage indices (0-based) to hold out from training. "
                             "Example: '2' for LOO at the middle stage, '1,3' for OTP-FM every-other, "
                             "'1' for LOO at the first interior stage. Held-out stages are still "
                             "evaluated at chained integration time.")
    parser.add_argument("--mm-si-sigma", type=float, default=0.05,
                        help="Stochastic-interpolant training noise for MM+SI methods, "
                             "interpreted as the TARGET BROWNIAN ARC LENGTH in radians on "
                             "the sphere (default: 0.05). Internally divided by sqrt(D-1) "
                             "to get the per-component scale passed to sphere_brownian_perturb. "
                             "Set to 0 to disable.")
    parser.add_argument("--mm-inf-sigma", type=float, default=0.0,
                        help="Inference-time SDE Brownian kick for all MM eval (default: 0 = "
                             "deterministic Euler). Set to e.g. 0.05 to add a tangent-space "
                             "Gaussian increment at each Euler step.")
    parser.add_argument("--mm-biharm-beta", type=float, default=0.3,
                        help="Biharmonic velocity blend factor for MM+BiharmonicVel "
                             "(default: 0.3). Blends SLERP velocity with log-map direction "
                             "to target, scaled to match SLERP velocity magnitude.")
    parser.add_argument("--mm-visualize", action="store_true",
                        help="Generate interactive PHATE trajectory visualization (first seed "
                             "only). Saves to trajectories_phate.html.")
    parser.add_argument("--stark-iters", type=int, default=50000,
                        help="Training iterations per method for the Stark experiment (default: 50000).")
    parser.add_argument("--stark-seeds", type=int, default=1,
                        help="Number of random seeds for the Stark experiment (default: 1).")
    parser.add_argument("--stark-K", type=str, default=None,
                        help="Comma-separated K values for the Stark experiment (default: 20,40,...,160).")
    args = parser.parse_args()

    if args.mode == "embryoid":
        main_embryoid(
            max_cells_per_stage=args.n_cells,
            ot_subsample=args.ot_subsample,
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            knn=args.knn,
            methods=args.methods,
            score_ncells=args.score_ncells,
            score_sigma=args.score_sigma,
        )
    elif args.mode == "diagnose":
        main_diagnose(ot_subsample=args.ot_subsample, knn=args.knn)
    elif args.mode == "conflict":
        main_conflict_sweep(ot_subsample=args.ot_subsample, knn=args.knn)
    elif args.mode == "mm":
        main_mm(
            n_iters=args.n_iters,
            batch_size=args.batch_size,
            ot_subsample=args.ot_subsample,
            score_iters=args.mm_score_iters,
            score_alpha=args.mm_alpha,
            score_net_sigma=args.mm_score_net_sigma,
            methods=args.methods,
            n_seeds=args.mm_seeds,
            holdout_stages=args.mm_holdout_stages,
            otpfm_holdout=args.mm_otpfm_holdout,
            si_sigma=args.mm_si_sigma,
            inf_sigma=args.mm_inf_sigma,
            biharm_beta=args.mm_biharm_beta,
            visualize=args.mm_visualize,
        )
    elif args.mode == "stark":
        K_values = None
        if args.stark_K:
            K_values = [int(x) for x in args.stark_K.split(",")]
        main_stark(
            K_values=K_values,
            n_iters=args.stark_iters,
            n_seeds=args.stark_seeds,
        )
    else:
        main_toy()


if __name__ == "__main__":
    main()
