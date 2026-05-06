"""
Evaluation metrics for comparing predicted and ground-truth cell distributions.
"""

import warnings

import numpy as np
import torch
from scipy import linalg

try:
    from ot.sliced import sliced_wasserstein_distance

    HAS_POT_SLICED = True
except ImportError:
    HAS_POT_SLICED = False


def _as_float_tensor(X):
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X).float()
    return X.float() if X.dtype != torch.float32 else X


def _sum_exp_sqdist(A, B, denominators, chunk_size=4096):
    """Sum exp(-||a-b||^2 / denom) over all pairs, in chunks.

    Chunks the cdist along axis 0 of A so the (chunk x |B|) distance matrix
    fits in GPU/CPU memory; for typical eval sizes (N <= ~5000) the whole
    thing usually fits in one chunk on a modern GPU.
    """
    total = A.new_tensor(0.0)
    denominators = [float(d) for d in denominators]
    for start in range(0, len(A), chunk_size):
        d2 = torch.cdist(A[start : start + chunk_size], B).square()
        for denom in denominators:
            total = total + torch.exp(-d2 / denom).sum()
    return total


def mmd_rbf(X, Y, sigma=None):
    """MMD^2 with RBF kernel. X, Y: (n, D) tensors."""
    X = _as_float_tensor(X)
    Y = _as_float_tensor(Y).to(X.device)
    if sigma is None:
        # Median heuristic
        all_pts = torch.cat([X, Y], dim=0)
        dists = torch.cdist(all_pts, all_pts)
        sigma = dists.median().item() / np.sqrt(2)
        sigma = max(sigma, 1e-3)

    n, m = len(X), len(Y)
    denom = 2 * sigma**2
    kxx_sum = _sum_exp_sqdist(X, X, [denom])
    kyy_sum = _sum_exp_sqdist(Y, Y, [denom])
    kxy_sum = _sum_exp_sqdist(X, Y, [denom])
    mmd2 = (
        (kxx_sum - n) / (n * (n - 1))
        + (kyy_sum - m) / (m * (m - 1))
        - 2 * kxy_sum / (n * m)
    )
    return mmd2.item()


def mmd_otpfm(X, Y, kernel_mul=2.0, kernel_num=5):
    """Multi-scale RBF MMD^2 matching the OTP-FM reference implementation.

    Reference: experiments/common/evaluation.py::compute_mmd in
    OTP-FM (Atanackovic et al.). Differences vs ``mmd_rbf`` (this file):

      * ``mmd_rbf`` uses a SINGLE Gaussian kernel with sigma chosen by the
        median heuristic on pairwise distances; the kernel is
        ``exp(-d^2 / (2 sigma^2))`` with ``sigma = median(d)/sqrt(2)``.
      * ``mmd_otpfm`` uses FIVE Gaussian kernels at scales
        ``b * 2^i`` for ``i in [0..4]`` summed together; the kernel is
        ``exp(-d^2 / b)`` and the central bandwidth ``b`` is the MEAN of
        squared pairwise distances divided by 2^(kernel_num // 2) = 4.

    Both are biased empirical MMD^2 estimators with self-pairs included
    in the within-set kernel means; the difference in bandwidth selection
    and number of kernels is what shifts held-out MMD^2 by ~5x between
    the two conventions on the OTP-FM EB benchmark.
    """
    X = _as_float_tensor(X)
    Y = _as_float_tensor(Y).to(X.device)

    source = X.reshape(X.shape[0], -1)
    target = Y.reshape(Y.shape[0], -1)
    n_source = source.shape[0]
    n_target = target.shape[0]
    n_total = n_source + n_target

    total = torch.cat([source, target], dim=0)

    # Bandwidth = mean of squared distances (excluding diagonal terms via
    # the n^2 - n divisor used by OTP-FM, although the diagonal is zero
    # so the sum is unchanged).
    total_sq_norm = total.square().sum(dim=1).sum()
    total_sum = total.sum(dim=0)
    pairwise_sq_sum = 2 * n_total * total_sq_norm - 2 * total_sum.square().sum()
    bandwidth = float(pairwise_sq_sum.item()) / max(n_total ** 2 - n_total, 1)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth = max(bandwidth, 1e-12)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    K_xx = _sum_exp_sqdist(source, source, bandwidth_list)
    K_yy = _sum_exp_sqdist(target, target, bandwidth_list)
    K_xy = _sum_exp_sqdist(source, target, bandwidth_list)
    # K_yx = sum_{i,j} k(y_i, x_j) is identical to K_xy = sum_{i,j} k(x_i, y_j)
    # for symmetric kernels (Gaussian d^2), so we reuse it instead of running
    # cdist a fourth time. The original OTP-FM reference computes both, but
    # the result is mathematically the same.

    # OTP-FM uses biased estimator: includes self-similarity (diagonal of
    # K_xx, K_yy). Match exactly so absolute values are comparable.
    return float(
        K_xx / (n_source * n_source)
        + K_yy / (n_target * n_target)
        - 2.0 * K_xy / (n_source * n_target)
    )


def w2(X, Y):
    """Exact Wasserstein-2 distance via POT's network simplex solver.

    Builds the full pairwise squared-Euclidean cost matrix between
    empirical samples X (n, D) and Y (m, D), solves the linear OT
    problem with uniform marginals (1/n, 1/m), and returns
    sqrt(<T, ||x-y||^2>) -- the standard W_2 distance, in the same
    units as ||x-y||. For n, m up to a few thousand this runs in
    seconds; beyond that fall back to swd / sinkhorn-W_2.
    """
    import ot

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)

    n, m = len(X), len(Y)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    M = np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2.0 * X @ Y.T
    M = np.clip(M, 0.0, None)  # guard tiny negatives from FP error
    cost = ot.emd2(a, b, M)
    return float(np.sqrt(max(cost, 0.0)))


def swd(
    X,
    Y,
    n_projections=50,
):
    """Sliced Wasserstein Distance using POT's reference implementation."""
    if not HAS_POT_SLICED:
        raise ImportError(
            "POT sliced Wasserstein support is required for SWD. "
            "Install/update `pot` in this environment."
        )

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    return float(sliced_wasserstein_distance(X, Y, n_projections=n_projections))


def fgd(
    X,
    Y,
    eps=1e-6,
):
    """Fréchet Gaussian Distance between two empirical sample clouds."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)

    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)
    sigma_x = np.cov(X, rowvar=False)
    sigma_y = np.cov(Y, rowvar=False)

    mu_x = np.atleast_1d(mu_x)
    mu_y = np.atleast_1d(mu_y)
    sigma_x = np.atleast_2d(sigma_x)
    sigma_y = np.atleast_2d(sigma_y)

    if mu_x.shape != mu_y.shape:
        raise ValueError("FGD requires equal feature dimensions for both sample sets")
    if sigma_x.shape != sigma_y.shape:
        raise ValueError("FGD requires equal covariance dimensions for both sample sets")

    mean_diff = mu_x - mu_y
    covmean = linalg.sqrtm(sigma_x.dot(sigma_y))
    if not np.isfinite(covmean).all():
        warnings.warn(
            f"FGD encountered a near-singular covariance product; adding {eps} to the diagonal",
            RuntimeWarning,
            stacklevel=2,
        )
        offset = np.eye(sigma_x.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_x + offset).dot(sigma_y + offset))

    if np.iscomplexobj(covmean):
        if not (
            np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
            or np.isclose(np.trace(covmean.imag) / np.trace(covmean.real), 0, atol=1e-3)
        ):
            warnings.warn(
                "FGD covariance square root has a non-negligible imaginary component; "
                "using the real part.",
                RuntimeWarning,
                stacklevel=2,
            )
        covmean = covmean.real

    trace_covmean = np.trace(covmean)
    fgd_sq = (
        mean_diff.dot(mean_diff)
        + np.trace(sigma_x)
        + np.trace(sigma_y)
        - 2 * trace_covmean
    )
    return float(np.sqrt(max(fgd_sq, 0.0)))
