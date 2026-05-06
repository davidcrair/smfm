"""Spectral premetric helpers for conditional flow targets on a graph manifold."""

import numpy as np
import torch

from surf.geometry.sphere import normalize_sphere, sphere_log_map
from surf.ot.costs import compute_global_biharmonic_embedding


class SpectralPremetric:
    """Spectral premetric on a subsampled training graph.

    The premetric lives in the global spectral embedding space built from the
    subsampled stage cells used during training.

    Two trajectory modes are supported:

    - ``spectral_decode`` (default): linearly interpolate in the global
      spectral embedding and decode each intermediate point back to the local
      graph manifold with a soft spectral kNN barycenter in ambient sphere
      coordinates. This keeps trajectories on the positive orthant and close
      to the observed data manifold.
    - ``graph_geodesic``: shortest path on the kNN graph between paired
      training cells, parameterized by graph arc length and interpolated as
      piecewise SLERP along path edges. This most directly follows the sampled
      data manifold.
    - ``spectral_decode_arclength``: decode a dense spectral polyline with a
      continuity-aware local decoder, then reparameterize by sphere arc length.
      This keeps the spectral premetric geometry but avoids the poor timing of
      raw global spectral decoding.
    - ``ode``: the legacy Chen-Lipman-style ambient ODE target built from a
      soft kNN extension of the spectral embedding. This remains available for
      ablations and debugging.
    """

    def __init__(
        self,
        stage_cells,
        *,
        knn_graph=15,
        n_eig=50,
        spectral_family="power",
        weight_power=0.5,
        diffusion_time=1.0,
        extension_k=64,
        softmax_beta=10.0,
        trajectory_mode="spectral_decode",
        decode_k=64,
        decode_beta=10.0,
        decode_chunk_size=64,
        velocity_fd_eps=0.02,
        time_cap=0.9,
        grad_norm_floor=0.05,
        max_drive_scale=50.0,
        eps=1e-6,
    ):
        self.device = stage_cells[0].device
        self.stage_cells = list(stage_cells)
        self.all_cells = torch.cat(stage_cells, dim=0)
        self.extension_k = min(int(extension_k), len(self.all_cells))
        self.softmax_beta = float(softmax_beta)
        if trajectory_mode not in (
            "spectral_decode",
            "spectral_decode_arclength",
            "graph_geodesic",
            "ode",
        ):
            raise ValueError(
                f"Unknown trajectory_mode={trajectory_mode!r}; "
                "expected 'spectral_decode', 'spectral_decode_arclength', "
                "'graph_geodesic', or 'ode'."
            )
        self.trajectory_mode = str(trajectory_mode)
        self.decode_k = min(int(decode_k), len(self.all_cells))
        self.decode_beta = float(decode_beta)
        self.decode_chunk_size = max(1, int(decode_chunk_size))
        self.velocity_fd_eps = float(velocity_fd_eps)
        self.time_cap = float(time_cap)
        self.grad_norm_floor = float(grad_norm_floor)
        self.max_drive_scale = None if max_drive_scale is None else float(max_drive_scale)
        self.eps = float(eps)
        self.spectral_family = spectral_family
        self.weight_power = float(weight_power)
        self.diffusion_time = float(diffusion_time)
        self.stage_sizes = [len(cells) for cells in self.stage_cells]
        self.stage_offsets = np.cumsum([0] + self.stage_sizes[:-1]).astype(np.int64)

        emb_per_stage = compute_global_biharmonic_embedding(
            stage_cells,
            knn=knn_graph,
            n_eig=n_eig,
            spectral_family=spectral_family,
            weight_power=weight_power,
            diffusion_time=diffusion_time,
        )
        self.stage_embeddings_np = [emb.astype("float32") for emb in emb_per_stage]
        self.stage_embeddings = [
            torch.from_numpy(emb).to(self.device)
            for emb in self.stage_embeddings_np
        ]
        self.all_embeddings = torch.cat(self.stage_embeddings, dim=0)
        self.all_embedding_norm_sq = (self.all_embeddings * self.all_embeddings).sum(
            dim=-1
        )
        # MPS is prone to out-of-memory failures from repeated topk/gather
        # temporaries in spectral decoding. The decoded target is used under
        # no_grad as a supervision target, so doing this kNN decode on CPU is
        # slower but much more memory-stable on Apple GPUs.
        self.decode_on_cpu = self.device.type == "mps"
        self.all_cells_decode = (
            self.all_cells.detach().cpu() if self.decode_on_cpu else self.all_cells
        )
        self.all_embeddings_decode = (
            self.all_embeddings.detach().cpu()
            if self.decode_on_cpu
            else self.all_embeddings
        )
        self.all_embedding_norm_sq_decode = (
            self.all_embedding_norm_sq.detach().cpu()
            if self.decode_on_cpu
            else self.all_embedding_norm_sq
        )
        self.graph_lengths = self._build_graph_lengths(self.all_cells, knn_graph)

    def _build_graph_lengths(self, all_cells, knn_graph):
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix

        all_np = all_cells.detach().cpu().numpy()
        N = len(all_np)
        if N <= 1:
            return csr_matrix((N, N), dtype=np.float64)

        n_neighbors = min(int(knn_graph) + 1, N)
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(all_np)
        dists, inds = nn.kneighbors(all_np)
        if n_neighbors == 1:
            return csr_matrix((N, N), dtype=np.float64)

        cos_sim = np.clip(1.0 - dists[:, 1:], -1 + 1e-6, 1 - 1e-6)
        arc = np.arccos(cos_sim)
        rows = np.repeat(np.arange(N), n_neighbors - 1)
        cols = inds[:, 1:].reshape(-1)
        data = arc.reshape(-1)
        graph = csr_matrix((data, (rows, cols)), shape=(N, N))
        return graph.maximum(graph.T).tocsr()

    def target_embeddings(self, stage_idx, local_idx):
        """Return spectral target embeddings for a batch of local indices."""
        if not torch.is_tensor(local_idx):
            local_idx = torch.as_tensor(local_idx, device=self.device, dtype=torch.long)
        else:
            local_idx = local_idx.to(device=self.device, dtype=torch.long)
        return self.stage_embeddings[stage_idx][local_idx]

    def _to_global_idx(self, stage_idx, local_idx):
        if torch.is_tensor(local_idx):
            local_np = local_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            local_np = np.asarray(local_idx, dtype=np.int64)
        return self.stage_offsets[int(stage_idx)] + local_np

    def stage_points(self, stage_idx, local_idx):
        """Return stored ambient sphere points for a batch of local indices."""
        if not torch.is_tensor(local_idx):
            local_idx = torch.as_tensor(local_idx, device=self.device, dtype=torch.long)
        else:
            local_idx = local_idx.to(device=self.device, dtype=torch.long)
        return self.stage_cells[stage_idx][local_idx]

    def clamp_time(self, t_local):
        return t_local.clamp(min=0.0, max=self.time_cap)

    def clamp_path_time(self, t_local):
        """Clamp direct decoded paths without applying the ODE singularity cap."""
        return t_local.clamp(min=0.0, max=1.0)

    def stage_pair_cost(self, src_stage_idx, tgt_stage_idx):
        """Return squared spectral cost between two stored stages."""
        from scipy.spatial.distance import cdist

        e0 = self.stage_embeddings_np[src_stage_idx]
        e1 = self.stage_embeddings_np[tgt_stage_idx]
        return cdist(e0, e1, metric="sqeuclidean").astype(np.float32, copy=False)

    def _soft_extension(self, x):
        sims = x @ self.all_cells.T
        top_sims, top_idx = torch.topk(sims, k=self.extension_k, dim=1)
        neigh_cells = self.all_cells[top_idx]
        neigh_emb = self.all_embeddings[top_idx]
        weights = torch.softmax(self.softmax_beta * top_sims, dim=1)
        phi_x = (weights.unsqueeze(-1) * neigh_emb).sum(dim=1)
        return phi_x, weights, neigh_cells, neigh_emb

    def _normalize_orthant(self, x):
        return normalize_sphere(x.clamp(min=self.eps))

    def _decode_from_spectral(self, emb):
        """Decode spectral coordinates back to the data manifold on the sphere."""
        decoded_chunks = []
        emb_decode = emb.detach().cpu() if self.decode_on_cpu else emb
        all_embeddings = self.all_embeddings_decode
        all_embedding_norm_sq = self.all_embedding_norm_sq_decode
        all_cells = self.all_cells_decode
        for emb_chunk in emb_decode.split(self.decode_chunk_size, dim=0):
            emb_norm_sq = (emb_chunk * emb_chunk).sum(dim=-1, keepdim=True)
            dist_sq = emb_norm_sq + all_embedding_norm_sq.unsqueeze(0) - 2.0 * (
                emb_chunk @ all_embeddings.T
            )
            dist_sq = dist_sq.clamp(min=0.0)
            top_dist_sq, top_idx = torch.topk(
                dist_sq,
                k=self.decode_k,
                dim=1,
                largest=False,
            )
            weights = torch.softmax(-self.decode_beta * top_dist_sq, dim=1)
            flat_idx = top_idx.reshape(-1)
            neigh_cells = all_cells.index_select(0, flat_idx).reshape(
                emb_chunk.shape[0],
                self.decode_k,
                all_cells.shape[1],
            )
            decoded_chunks.append((weights.unsqueeze(-1) * neigh_cells).sum(dim=1))
        decoded = torch.cat(decoded_chunks, dim=0)
        decoded = self._normalize_orthant(decoded)
        return decoded.to(self.device) if self.decode_on_cpu else decoded

    def _decode_from_spectral_continuous(self, emb, prev_point):
        """Decode spectral coordinates with a local continuity penalty.

        The raw spectral decoder can jump to high-density regions because it
        only sees spectral distance. This decoder first restricts to spectral
        neighbors, then softly balances spectral closeness with sphere-arc
        closeness to the previous decoded point.
        """
        decoded_chunks = []
        emb_decode = emb.detach().cpu() if self.decode_on_cpu else emb
        prev_decode = prev_point.detach().cpu() if self.decode_on_cpu else prev_point
        all_embeddings = self.all_embeddings_decode
        all_embedding_norm_sq = self.all_embedding_norm_sq_decode
        all_cells = self.all_cells_decode
        for emb_chunk, prev_chunk in zip(
            emb_decode.split(self.decode_chunk_size, dim=0),
            prev_decode.split(self.decode_chunk_size, dim=0),
        ):
            emb_norm_sq = (emb_chunk * emb_chunk).sum(dim=-1, keepdim=True)
            dist_sq = emb_norm_sq + all_embedding_norm_sq.unsqueeze(0) - 2.0 * (
                emb_chunk @ all_embeddings.T
            )
            dist_sq = dist_sq.clamp(min=0.0)
            top_dist_sq, top_idx = torch.topk(
                dist_sq,
                k=self.decode_k,
                dim=1,
                largest=False,
            )
            flat_idx = top_idx.reshape(-1)
            neigh_cells = all_cells.index_select(0, flat_idx).reshape(
                emb_chunk.shape[0],
                self.decode_k,
                all_cells.shape[1],
            )

            cos_prev = (neigh_cells * prev_chunk.unsqueeze(1)).sum(dim=-1)
            cos_prev = cos_prev.clamp(-1 + 1e-6, 1 - 1e-6)
            arc_sq = torch.acos(cos_prev).square()

            spectral_scale = top_dist_sq.median(dim=1, keepdim=True).values.clamp(min=1e-8)
            arc_scale = arc_sq.median(dim=1, keepdim=True).values.clamp(min=1e-8)
            score = top_dist_sq / spectral_scale + arc_sq / arc_scale
            weights = torch.softmax(-self.decode_beta * score, dim=1)
            decoded_chunks.append((weights.unsqueeze(-1) * neigh_cells).sum(dim=1))

        decoded = torch.cat(decoded_chunks, dim=0)
        decoded = self._normalize_orthant(decoded)
        return decoded.to(self.device) if self.decode_on_cpu else decoded

    def _spectral_decode_position(self, src_emb, target_emb, t_local, src_point=None, target_point=None):
        tau = self.clamp_path_time(t_local)
        emb_t = (1.0 - tau) * src_emb + tau * target_emb
        z_t = self._decode_from_spectral(emb_t)
        if src_point is not None:
            z_t = torch.where(tau <= self.eps, src_point, z_t)
        if target_point is not None:
            z_t = torch.where(tau >= 1.0 - self.eps, target_point, z_t)
        return z_t

    def _spectral_decode_velocity(
        self,
        src_emb,
        target_emb,
        t_local,
        z_t=None,
        src_point=None,
        target_point=None,
    ):
        tau = self.clamp_path_time(t_local)
        dt = float(self.velocity_fd_eps)
        tau_minus = self.clamp_path_time(tau - dt)
        tau_plus = self.clamp_path_time(tau + dt)
        if z_t is None:
            z_t = self._spectral_decode_position(
                src_emb, target_emb, tau,
                src_point=src_point, target_point=target_point,
            )
        z_minus = self._spectral_decode_position(
            src_emb, target_emb, tau_minus,
            src_point=src_point, target_point=target_point,
        )
        z_plus = self._spectral_decode_position(
            src_emb, target_emb, tau_plus,
            src_point=src_point, target_point=target_point,
        )
        v_forward = sphere_log_map(z_t, z_plus) / (tau_plus - tau).clamp(min=self.eps)
        v_backward = -sphere_log_map(z_t, z_minus) / (tau - tau_minus).clamp(min=self.eps)
        use_forward = (tau - tau_minus) <= self.eps
        use_backward = (tau_plus - tau) <= self.eps
        v_local = torch.where(
            use_forward,
            v_forward,
            torch.where(use_backward, v_backward, 0.5 * (v_forward + v_backward)),
        )
        # Keep the local-time denominator explicit for endpoint one-sided cases.
        v_local = torch.where((tau_plus - tau_minus) <= self.eps, torch.zeros_like(v_local), v_local)
        return v_local - (v_local * z_t).sum(dim=-1, keepdim=True) * z_t

    def _spectral_arclength_polyline(
        self,
        src_emb,
        target_emb,
        src_point,
        target_point,
        n_steps,
    ):
        n_nodes = max(int(n_steps) + 1, 2)
        tau_grid = torch.linspace(
            0.0,
            1.0,
            n_nodes,
            device=self.device,
            dtype=src_emb.dtype,
        )
        points = [src_point]
        prev = src_point
        for tau in tau_grid[1:-1]:
            emb_tau = (1.0 - tau) * src_emb + tau * target_emb
            prev = self._decode_from_spectral_continuous(emb_tau, prev)
            points.append(prev)
        points.append(target_point)

        path = torch.stack(points, dim=1)
        cos_edges = (path[:, :-1] * path[:, 1:]).sum(dim=-1)
        cos_edges = cos_edges.clamp(-1 + 1e-6, 1 - 1e-6)
        seg_len = torch.acos(cos_edges)
        cumlen = torch.cat(
            [
                torch.zeros(len(path), 1, device=self.device, dtype=path.dtype),
                torch.cumsum(seg_len, dim=1),
            ],
            dim=1,
        )
        return path, cumlen

    def _sample_spherical_polyline(self, path, cumlen, t_local):
        tau = self.clamp_path_time(t_local)
        total_len = cumlen[:, -1:].clamp(min=self.eps)
        target_len = tau * total_len
        seg_idx = (cumlen[:, 1:] <= target_len).sum(dim=1)
        seg_idx = seg_idx.clamp(max=path.shape[1] - 2)

        gather_idx = seg_idx.view(-1, 1, 1).expand(-1, 1, path.shape[-1])
        start = torch.gather(path, 1, gather_idx).squeeze(1)
        end = torch.gather(path, 1, gather_idx + 1).squeeze(1)
        len_idx = seg_idx.view(-1, 1)
        seg_start_len = torch.gather(cumlen, 1, len_idx)
        seg_end_len = torch.gather(cumlen, 1, len_idx + 1)
        cur_seg_len = (seg_end_len - seg_start_len).clamp(min=self.eps)
        u = ((target_len - seg_start_len) / cur_seg_len).clamp(0.0, 1.0)

        cos_omega = (start * end).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        w0 = torch.sin((1.0 - u) * omega) / sin_omega
        w1 = torch.sin(u * omega) / sin_omega
        z_t = self._normalize_orthant(w0 * start + w1 * end)

        v_u = omega * (
            -torch.cos((1.0 - u) * omega) / sin_omega * start
            + torch.cos(u * omega) / sin_omega * end
        )
        v_u = v_u - (v_u * z_t).sum(dim=-1, keepdim=True) * z_t
        v_local = v_u * (total_len / cur_seg_len)
        inactive = total_len <= self.eps
        v_local = torch.where(inactive, torch.zeros_like(v_local), v_local)
        return z_t, v_local

    def _spectral_decode_arclength_position_velocity(
        self,
        src_emb,
        target_emb,
        t_local,
        src_point,
        target_point,
        n_steps,
    ):
        path, cumlen = self._spectral_arclength_polyline(
            src_emb,
            target_emb,
            src_point,
            target_point,
            n_steps=n_steps,
        )
        return self._sample_spherical_polyline(path, cumlen, t_local)

    def _arc_between_nodes(self, global_idx_a, global_idx_b):
        pts_a = self.all_cells[torch.as_tensor(global_idx_a, device=self.device, dtype=torch.long)]
        pts_b = self.all_cells[torch.as_tensor(global_idx_b, device=self.device, dtype=torch.long)]
        cos = (pts_a * pts_b).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.acos(cos).detach().cpu().numpy().astype(np.float64, copy=False)

    def _reconstruct_path(self, predecessors, src_global, tgt_global):
        no_pred = -9999
        if src_global == tgt_global:
            return np.asarray([src_global], dtype=np.int64)
        path = [int(tgt_global)]
        cur = int(tgt_global)
        while cur != int(src_global):
            cur = int(predecessors[cur])
            if cur == no_pred:
                return np.asarray([int(src_global), int(tgt_global)], dtype=np.int64)
            path.append(cur)
            if len(path) > len(predecessors) + 1:
                return np.asarray([int(src_global), int(tgt_global)], dtype=np.int64)
        path.reverse()
        return np.asarray(path, dtype=np.int64)

    def prepare_interval_paths(self, src_stage_idx, tgt_stage_idx, src_local_idx, tgt_local_idx):
        """Precompute shortest-path metadata for a batch of paired endpoints."""
        from scipy.sparse.csgraph import shortest_path

        src_global = self._to_global_idx(src_stage_idx, src_local_idx)
        tgt_global = self._to_global_idx(tgt_stage_idx, tgt_local_idx)
        n_pairs = len(src_global)
        paths = [None] * n_pairs
        by_source = {}
        for pair_idx, src in enumerate(src_global):
            by_source.setdefault(int(src), []).append(pair_idx)

        for src, pair_indices in by_source.items():
            _, predecessors = shortest_path(
                self.graph_lengths,
                directed=False,
                indices=int(src),
                return_predecessors=True,
                unweighted=False,
            )
            for pair_idx in pair_indices:
                tgt = int(tgt_global[pair_idx])
                nodes = self._reconstruct_path(predecessors, int(src), tgt)
                if len(nodes) <= 1:
                    cumlen = np.asarray([0.0], dtype=np.float64)
                    total_length = 0.0
                else:
                    edge_lengths = self._arc_between_nodes(nodes[:-1], nodes[1:])
                    cumlen = np.concatenate(
                        [np.asarray([0.0], dtype=np.float64), np.cumsum(edge_lengths)]
                    )
                    total_length = float(cumlen[-1])
                paths[pair_idx] = {
                    "nodes": nodes,
                    "cumlen": cumlen,
                    "total_length": total_length,
                }
        return paths

    def _graph_geodesic_position_velocity(self, path_cache, pair_indices, t_local):
        if torch.is_tensor(pair_indices):
            pair_idx_np = pair_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            pair_idx_np = np.asarray(pair_indices, dtype=np.int64)
        tau_np = self.clamp_time(t_local).detach().cpu().numpy().reshape(-1)

        start_nodes = np.empty(len(pair_idx_np), dtype=np.int64)
        end_nodes = np.empty(len(pair_idx_np), dtype=np.int64)
        seg_u = np.zeros(len(pair_idx_np), dtype=np.float32)
        seg_len = np.ones(len(pair_idx_np), dtype=np.float32)
        total_len = np.zeros(len(pair_idx_np), dtype=np.float32)
        active = np.zeros(len(pair_idx_np), dtype=bool)

        for row, pair_idx in enumerate(pair_idx_np):
            meta = path_cache[int(pair_idx)]
            nodes = meta["nodes"]
            cumlen = meta["cumlen"]
            total = float(meta["total_length"])
            total_len[row] = total
            if len(nodes) <= 1 or total <= self.eps:
                start_nodes[row] = int(nodes[0])
                end_nodes[row] = int(nodes[0])
                continue

            s = float(tau_np[row]) * total
            seg = int(np.searchsorted(cumlen[1:], s, side="right"))
            seg = min(max(seg, 0), len(nodes) - 2)
            seg_start = float(cumlen[seg])
            seg_stop = float(cumlen[seg + 1])
            cur_seg_len = max(seg_stop - seg_start, self.eps)
            start_nodes[row] = int(nodes[seg])
            end_nodes[row] = int(nodes[seg + 1])
            seg_len[row] = cur_seg_len
            seg_u[row] = float(np.clip((s - seg_start) / cur_seg_len, 0.0, 1.0))
            active[row] = True

        start_t = self.all_cells[torch.as_tensor(start_nodes, device=self.device, dtype=torch.long)]
        end_t = self.all_cells[torch.as_tensor(end_nodes, device=self.device, dtype=torch.long)]
        u_t = torch.as_tensor(seg_u[:, None], device=self.device, dtype=torch.float32)
        seg_len_t = torch.as_tensor(seg_len[:, None], device=self.device, dtype=torch.float32)
        total_len_t = torch.as_tensor(total_len[:, None], device=self.device, dtype=torch.float32)
        active_t = torch.as_tensor(active[:, None], device=self.device)

        cos_omega = (start_t * end_t).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        w0 = torch.sin((1.0 - u_t) * omega) / sin_omega
        w1 = torch.sin(u_t * omega) / sin_omega
        z_t = self._normalize_orthant(w0 * start_t + w1 * end_t)
        z_t = torch.where(active_t, z_t, start_t)

        v_u = omega * (
            -torch.cos((1.0 - u_t) * omega) / sin_omega * start_t
            + torch.cos(u_t * omega) / sin_omega * end_t
        )
        v_u = v_u - (v_u * z_t).sum(dim=-1, keepdim=True) * z_t
        v_local = v_u * (total_len_t / seg_len_t.clamp(min=self.eps))
        v_local = torch.where(active_t, v_local, torch.zeros_like(v_local))
        return z_t, v_local

    def distance_and_gradient(self, x, target_emb):
        """Compute d(x, y) and tangent grad_x d(x, y) for target embedding y."""
        phi_x, weights, neigh_cells, neigh_emb = self._soft_extension(x)
        delta = phi_x - target_emb
        d_sq = (delta * delta).sum(dim=-1, keepdim=True)
        d = torch.sqrt(d_sq + self.eps)

        delta_dot_e = (neigh_emb * delta.unsqueeze(1)).sum(dim=-1)
        weighted_cells = (weights.unsqueeze(-1) * neigh_cells).sum(dim=1)
        term1 = ((weights * delta_dot_e).unsqueeze(-1) * neigh_cells).sum(dim=1)
        delta_dot_phi = (delta * phi_x).sum(dim=-1, keepdim=True)
        grad_half_d_sq = self.softmax_beta * (term1 - delta_dot_phi * weighted_cells)
        grad_d = grad_half_d_sq / d
        grad_tan = grad_d - (grad_d * x).sum(dim=-1, keepdim=True) * x
        return d, grad_tan

    def distance(self, x, target_emb):
        """Compute d(x, y) without returning the gradient."""
        phi_x, _, _, _ = self._soft_extension(x)
        delta = phi_x - target_emb
        return torch.sqrt((delta * delta).sum(dim=-1, keepdim=True) + self.eps)

    def field(self, x, target_emb, t_local):
        """Chen-Lipman minimal-norm premetric field with kappa(t)=1-t."""
        tau = self.clamp_time(t_local)
        d, grad_tan = self.distance_and_gradient(x, target_emb)
        grad_norm = grad_tan.norm(dim=-1, keepdim=True).clamp(min=self.grad_norm_floor)
        drive_scale = d / grad_norm.square()
        if self.max_drive_scale is not None:
            drive_scale = drive_scale.clamp(max=self.max_drive_scale)
        return -(drive_scale / (1.0 - tau)) * grad_tan

    def integrate(self, x0, target_emb, t_local, n_steps=8):
        """Integrate the legacy ODE target x'(t)=u_t(x|y) from 0 to t_local."""
        x = x0.clone()
        cur_t = torch.zeros_like(t_local)
        base_dt = torch.full_like(t_local, 1.0 / max(int(n_steps), 1))
        t_goal = self.clamp_time(t_local)

        for _ in range(max(int(n_steps), 1)):
            next_t = torch.minimum(cur_t + base_dt, t_goal)
            dt = next_t - cur_t
            active = dt.squeeze(-1) > 0
            if not bool(active.any()):
                break

            mid_t = cur_t + 0.5 * dt
            u_mid = self.field(x, target_emb, mid_t)
            x_next = normalize_sphere(x + dt * u_mid)
            x = torch.where(active.unsqueeze(-1), x_next, x)
            cur_t = next_t

        return x

    def sample_trajectory(
        self,
        src_stage_idx,
        src_local_idx,
        tgt_stage_idx,
        tgt_local_idx,
        t_local,
        *,
        n_steps=8,
        path_cache=None,
        pair_indices=None,
    ):
        """Return the current premetric position and local-time velocity."""
        if self.trajectory_mode in ("spectral_decode", "spectral_decode_arclength"):
            src_emb = self.target_embeddings(src_stage_idx, src_local_idx)
            target_emb = self.target_embeddings(tgt_stage_idx, tgt_local_idx)
            src_point = self.stage_points(src_stage_idx, src_local_idx)
            target_point = self.stage_points(tgt_stage_idx, tgt_local_idx)
            if self.trajectory_mode == "spectral_decode_arclength":
                return self._spectral_decode_arclength_position_velocity(
                    src_emb,
                    target_emb,
                    t_local,
                    src_point,
                    target_point,
                    n_steps=n_steps,
                )
            z_t = self._spectral_decode_position(
                src_emb, target_emb, t_local,
                src_point=src_point, target_point=target_point,
            )
            v_local = self._spectral_decode_velocity(
                src_emb, target_emb, t_local, z_t=z_t,
                src_point=src_point, target_point=target_point,
            )
            return z_t, v_local
        if self.trajectory_mode == "graph_geodesic":
            if path_cache is None or pair_indices is None:
                raise ValueError(
                    "graph_geodesic trajectories require precomputed path_cache and "
                    "pair_indices; call sample_trajectory(..., path_cache=..., pair_indices=...)."
                )
            return self._graph_geodesic_position_velocity(path_cache, pair_indices, t_local)

        x0 = self.stage_points(src_stage_idx, src_local_idx)
        target_emb = self.target_embeddings(tgt_stage_idx, tgt_local_idx)
        z_t = self.integrate(x0, target_emb, t_local, n_steps=n_steps)
        v_local = self.field(z_t, target_emb, t_local)
        return z_t, v_local


BiharmonicPremetric = SpectralPremetric
