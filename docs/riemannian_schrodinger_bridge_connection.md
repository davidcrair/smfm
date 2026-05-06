# Riemannian Schrödinger Bridge — connection to this project

Reference doc for the writeup. Captures every relevant paper, the conceptual
chain that connects them to our spectral OT cost family, suggested framings,
and honest assessments of novelty.

## TL;DR

Our spectral OT cost in multi-marginal flow matching is best viewed as a
**Schrödinger-bridge-inspired manifold cost surrogate**, not as a literal
Schrödinger bridge solver. A Riemannian Schrödinger bridge uses a Brownian
reference process on a manifold; in the small-noise / short-time limit, the
endpoint cost is governed by squared geodesic distance. We approximate
manifold geometry with graph-Laplacian spectral distances and use those
distances only in the OT coupling step, while the flow-matching paths remain
deterministic (Euclidean chords or Fisher-sphere geodesics).

Thus the honest claim is: we borrow the SB/OT principle that endpoint
couplings should respect intrinsic geometry, instantiate that principle with
a parameterized graph-spectral cost family, and find that the best spectral
regime is dataset-dependent.

## The conceptual chain

```
Riemannian Schrödinger bridge with Brownian reference
   |
   |  small-noise / short-time endpoint asymptotics give squared geodesic cost
   v
Squared geodesic distance on the data manifold
   |
   |  Manifold Laplace-Beltrami operator (Belkin-Niyogi 2008
   |  consistency: kNN graph Laplacian -> Laplace-Beltrami as N -> infty
   |  with the right scaling)
   v
Graph Laplacian as a discrete proxy for manifold heat/Laplace-Beltrami geometry
   |
   |  spectral filters define intrinsic graph costs
   v
Family of OT ground costs:
  alpha = 0   : truncated spectral embedding distance
  alpha = 1   : commute-time / effective-resistance distance
  alpha = 2   : biharmonic distance
  heat weights: diffusion distance / heat-kernel geometry
   |
   |  Use as cost_fn in (multi-marginal) flow matching's OT coupling step
   v
Deterministic multi-marginal FM with manifold-aware couplings
```

Each arrow has a specific paper anchoring it, but the composition is an
informal motivation rather than a proven convergence theorem for our method.
See the bibliography below.

## Key papers (with what each gives us)

### Schrödinger bridge theory

- **Léonard 2014** *"A survey of the Schrödinger problem and some of its
  connections with optimal transport"* (arXiv:1308.0215, Discrete Contin.
  Dyn. Syst.). Foundational survey. §1.5–1.6 derives the ε→0 connection
  between Schrödinger problems and optimal transport; for Brownian reference
  processes on a manifold, the associated endpoint cost is governed by
  geodesic distance. §3 connects discrete Markov-chain / graph Schrödinger
  problems to the same general framework. **Use this for the theoretical
  anchor that SB, Brownian reference processes, and intrinsic OT costs are
  linked. Do not overstate this as proving our spectral cost is an SB cost.**

- **Mikami 2004** *"Monge's problem with a quadratic cost by the zero-noise
  limit of h-path processes"* (Probab. Theory Relat. Fields). Establishes
  the small-ε limit rigorously. Cite if you want a more formal handle on
  the limit than Léonard's survey.

- **Conforti & Léonard 2017** *"Reciprocal classes of random walks on
  graphs"* (arXiv:1601.07090, Stoch. Proc. Appl.). Discrete/graph version
  of the SB problem. Random walks are the discrete analog of Brownian
  motion; the heat semigroup `e^{-tL}` plays the role of Brownian
  transition kernels. **Cite this when justifying graph Laplacian as the
  discrete approximation of manifold Brownian motion.**

### Modern ML — Schrödinger bridge & flow matching

- **De Bortoli, Thornton, Heng, Doucet 2021** *"Diffusion Schrödinger Bridge
  with Applications to Score-Based Generative Modeling"* (arXiv:2106.01357,
  NeurIPS 2021). The canonical DSB paper. Uses Iterative Proportional
  Fitting (IPF) with simulated SDE paths to approximate the SB. Cite for
  the modern ML framing that connects SB to score-based / diffusion /
  flow-matching style training.

- **Chen, Liu, Theodorou 2022** *"Likelihood Training of Schrödinger Bridge
  using FB-SDEs"* (arXiv:2110.11291, ICLR 2022). FB-SDE-based SB training.
  Adjacent prior art.

- **Tong, Malkin, Fatras, Atanackovic, Zhang, Huguet, Wolf, Bengio 2024**
  *"Simulation-free Schrödinger bridges via score and flow matching"*
  (arXiv:2307.03672, AISTATS 2024). **This is the most directly relevant
  paper for our framing.** Key result: with Brownian-bridge interpolants
  between OT-coupled endpoints + a flow-matching loss + a score-matching
  loss, training is **simulation-free** and targets the SB without simulating
  the learned stochastic process. This is the right paper to cite when
  discussing what would be required to turn our deterministic setup into an
  actual SB method: entropic OT, noisy/Brownian bridge interpolants, and score
  matching. Our current contribution is more modest: replace the usual
  Euclidean OT cost in deterministic FM coupling construction with a
  manifold-aware spectral cost.

- **Tong et al. 2024 (TMLR)** *"Improving and generalizing flow-based
  generative models with minibatch optimal transport"* (OT-CFM). The
  immediate predecessor to [SF]^2M and to our work — it establishes
  OT-coupled flow matching as a generalization of CFM. Cite as the
  foundational OT-flow-matching paper.

- **Tong, Huguet, Natik, MacDonald, Kuchroo, Coifman, Wolf, Krishnaswamy
  2020** *"TrajectoryNet: A Dynamic Optimal Transport Network for Modeling
  Cellular Dynamics"* (ICML 2020). Single-cell trajectory inference with
  dynamic OT and neural ODEs. **Cite as a baseline / predecessor in the
  trajectory-inference paragraph.** They use the same EB benchmark we do.

### Riemannian Schrödinger bridge

- **Thornton, De Bortoli, Heng, Doucet 2022** *"Riemannian Diffusion
  Schrödinger Bridge"* (arXiv:2207.03024). Canonical Riemannian DSB paper.
  Generalizes DSB to **Riemannian manifolds** (closed-form sphere, torus).
  Reference measure = Brownian motion on the manifold, IPF run on the
  manifold. Evaluated on Earth-event data on $S^2$. **Cite as the most
  direct neighbor of our work**: they target the same manifold-aware SB
  but require a known manifold and use simulation; we use a kNN-graph
  approximation of an unknown data manifold and a simulation-free flow-
  matching frame.

- **Chen & Lipman 2024** *"Flow Matching on General Geometries"*
  (arXiv:2302.03660, ICLR 2024). Riemannian flow matching. No SB/OT focus,
  but provides the flow-matching primitive on manifolds. Cite if discussing
  sphere flow-matching specifically.

- **Lou, Lopez, Aliakbarpour, Bao, Hong, Liu 2024** *"Reflected Schrödinger
  Bridge"* (arXiv:2401.03228). SB on bounded domains.

- **Shi, De Bortoli, Campbell, Doucet 2023** *"Diffusion Schrödinger Bridge
  Matching"* (arXiv:2303.16852, NeurIPS 2023). Iterative Markovian Fitting
  (IMF) for SB. Not specifically Riemannian but extends the matching frame.

### Spectral / manifold-aware costs

- **Coifman & Lafon 2006** *"Diffusion maps"* (Applied and Computational
  Harmonic Analysis). Foundational paper for diffusion-distance-style
  spectral coordinates. Cite as the source of the diffusion-distance
  weighting `e^{-2tλ}`, which is the cleanest spectral link to heat-kernel /
  Brownian diffusion time.

- **Belkin & Niyogi 2008** *"Towards a theoretical foundation for
  Laplacian-based manifold methods"* (J. Comput. System Sci.). Establishes
  that the **kNN-graph normalized Laplacian converges to the manifold
  Laplace-Beltrami operator** as $N \to \infty$ with appropriate kernel
  bandwidth scaling. **Cite as the formal justification that our kNN-graph
  spectrum approximates manifold structure.**

- **Lipman, Rustamov, Funkhouser 2010** *"Biharmonic Distance"* (ACM TOG).
  Defines biharmonic distance as a spectral distance with $1/\lambda^2$
  weighting — i.e., $\alpha = 2$ in our squared-distance convention. Cite as
  the source of the biharmonic-distance regime.

- **Klein & Randić 1993** *"Resistance distance"* (J. Math. Chem.) and
  **Doyle & Snell 1984** *"Random walks and electric networks"*. Define
  effective-resistance / commute-time distance with $1/\lambda$ weighting —
  i.e., $\alpha = 1$ in our family.

- **Solomon, Rustamov, Guibas, Butscher 2014** *"Earth Mover's Distance on
  the Heat Kernel"* (ACM TOG). Earlier work using heat-kernel-based OT on
  meshes. Cite as prior art for spectral OT in non-trajectory settings.

- **Cuturi 2013** *"Sinkhorn distances"* (NeurIPS). Foundational entropic
  OT. Cite for the practical OT solver we use.

### Trajectory inference / single-cell

- **Schiebinger et al. 2019** *"Optimal-transport analysis of single-cell
  gene expression identifies developmental trajectories in reprogramming"*
  (Cell). The WOT (Waddington Optimal Transport) paper.

- **Moon et al. 2019** *"Visualizing structure and transitions in
  high-dimensional biological data"* (Nat. Biotechnol.). The PHATE paper +
  the Embryoid Body benchmark dataset. **The EB data we use comes from
  here**, specifically the preprocessed `eb_velocity_v5.npz` shipped with
  TrajectoryNet.

- **Tong et al. 2020** TrajectoryNet (cited above). EB at 100 PCs.

- **Huguet et al. 2022** *"Manifold Interpolating Optimal-Transport Flows
  for Trajectory Inference"* (arXiv:2206.14928, NeurIPS 2022). MIOFlow.
  **Closest prior work to our spectral OT idea**: uses a manifold-aware
  ground cost (PHATE-derived diffusion potential) for OT in trajectory
  inference. Differences from ours:
  - 2-marginal (consecutive pairs) vs our multi-marginal.
  - One fixed cost (PHATE potential) vs our parameterized $\alpha$-sweep.
  - Continuous-normalizing-flow framework vs flow matching.

- **Kapusniak, Atanackovic, Bose, Sano, Wolf, Bengio, Tong 2024** *"Metric
  Flow Matching for Smooth Interpolations on the Data Manifold"*
  (arXiv:2405.14780, NeurIPS 2024). MFM. Uses a metric (RBF / biharmonic)
  to define the conditional **path** during training, not the OT coupling.
  Different axis than ours. Cite as related but distinct.

- **Atanackovic, Theodoropoulos, Allen-Zhu, Tong, Bengio** *"Multimarginal
  flow matching with optimal transport potentials"* (OTP-FM, ICML 2026).
  Multi-marginal CFM with a curriculum on potential strength. **The most
  direct multi-marginal-flow-matching neighbor.** They use Euclidean OT;
  our spectral OT is the cost-side modification.

- **Rohbeck et al. 2025** *"MMFM: Modeling Complex System Dynamics with
  Flow Matching Across Time and Conditions"* (ICLR 2025). Multi-marginal
  flow matching with conditional dynamics. They use Euclidean OT and
  evaluate on synthetic + sciPlex T-cell + Beijing PM2.5. Cite alongside
  OTP-FM.

- **Davis, Lobo, Pegolotti, Rampášek, Bose 2024** *"Fisher Flow Matching
  for Generative Modeling over Discrete Data"* (arXiv:2405.14664, NeurIPS
  2024). Fisher Flow Matching on the simplex via the Fisher-Rao /
  positive-orthant-of-sphere geometry. **Source of our sphere encoding
  pipeline.** They do source -> target generative modeling (2-marginal);
  we extend to multi-marginal trajectory inference.

- **Shen, Wang, Liu 2024** *"Multi-marginal Schrödinger Bridges with
  Iterative Reference Refinement"* (arXiv:2408.06277). SB-IRR. Source of
  our GoM dataset. They iterate the reference process; we fix the cost and
  parameterize it with $\alpha$.

- **Theodoropoulos et al. 2025** *"Momentum Multi-Marginal Schrödinger
  Bridge Matching"* (arXiv:2506.10168). 3MSBM. Multi-marginal SB matching
  with momentum-augmented dynamics. Closest direct comparison if you want
  to cite an SB-flavored neighbor in the trajectory setting.

## Suggested writeup framings

### Tight related-work paragraph (≈80 words)

> *"[SF]$^2$M (Tong et al., 2024) shows that flow matching with OT-coupled
> endpoints and Brownian-bridge interpolants gives a simulation-free
> approximation of the Schrödinger bridge when paired with score matching and
> entropy-regularized OT. Our method does not solve a stochastic SB, but uses
> the same principle that endpoint couplings should respect the geometry of
> the reference space. For Brownian reference processes, the small-noise OT
> limit is governed by intrinsic geodesic distance (Léonard, 2014); on an
> unknown data manifold, graph Laplacian spectra provide discrete intrinsic
> geometry surrogates, including diffusion, resistance, and biharmonic
> distances (Coifman \& Lafon, 2006; Lipman et al., 2010). We use this
> spectral family as the OT ground cost in multi-marginal flow matching on
> the Fisher-Rao sphere (Davis et al., 2024)."*

### How we differ from Riemannian DSB (Thornton 2022)

| Axis | Riemannian DSB | Our work |
|---|---|---|
| Problem | 2-marginal source -> target | Multi-marginal trajectory inference |
| Manifold | Closed-form (S^2, torus) | Unknown — kNN-graph approximation |
| Reference process | Explicit Brownian motion on manifold | No explicit stochastic reference; spectral cost is an intrinsic-geometry surrogate |
| Algorithm | IPF + simulated manifold SDE paths | Deterministic flow matching with parameterized OT cost |
| Manifold encoding | The SDE simulation | The OT ground cost |
| Encoding strength | Fixed by Brownian reference and manifold metric | Tunable via spectral filter sweep |
| Eval | Sphere/torus toys | Real biological multi-marginal trajectory data |

### How we differ from MIOFlow (Huguet 2022)

| Axis | MIOFlow | Our work |
|---|---|---|
| OT cost | Single (PHATE diffusion potential) | Parameterized family ($\alpha$-sweep) |
| Marginals | 2-marginal | Multi-marginal |
| Geometry | CNF in PCA latent | Multi-marginal flow matching on Fisher-Rao sphere |
| Story | "Manifold cost helps" | "Optimal manifold-encoding strength is dataset-dependent" |

## Key theoretical observations from our analysis

1. **Disconnected-graph failure mode.** When the kNN graph splits into
   $k$ components, the normalized Laplacian has multiple near-zero
   eigenvalues. Since our implementation skips only the first trivial
   eigenvector and then applies inverse-power weights, component-indicator
   eigenvectors can dominate the cost. In the idealized separable-cost limit,
   cross-component OT may become underdetermined and resemble stratified or
   random matching, but this should be treated as a diagnostic/failure mode,
   not as a proven graceful degeneration. See Forrow et al. 2019
   *"Statistical OT via factored couplings"* for the stratified-OT
   formalization.

2. **Euclidean OT is outside the family.** No choice of $\alpha$ in the
   $1/\lambda^\alpha$ kNN-Laplacian-bottom-eigenvector family recovers
   Euclidean distance. The kNN-Laplacian eigenvectors converge to
   Laplace-Beltrami eigenfunctions on the data manifold — an *intrinsic*
   geometric object, not the ambient Euclidean. To recover Euclidean
   exactly via a spectral construction you'd need classical MDS / PCA: a
   complete graph with Gram-matrix weights, the **top** eigenvalues, and
   weight $w(\lambda) = \lambda$ — the opposite end of every axis. So
   Euclidean OT is a genuinely distinct ground cost (not a special case),
   which is why we treat it as a separate baseline.

3. **Adding stochasticity moves us closer to true Riemannian DSB.** The
   current setup is deterministic FM with manifold-aware couplings, best
   interpreted as an OT-limit / SB-inspired approximation rather than an SB
   solver. Adding entropic OT coupling, sphere-Brownian-bridge interpolants,
   and Riemannian score matching ([SF]$^2$M; sphere score matching) would
   move the method toward a finite-$\varepsilon$ Riemannian SB. Out of scope
   for this paper; valid future work.

4. **Convergence stack.** Strict justification of "our spectral cost is the
   discrete analog of a Riemannian SB endpoint cost" would require composing:
   - kNN-graph Laplacian -> manifold Laplace-Beltrami (Belkin-Niyogi
     consistency, requires $N \to \infty$ with specific scaling).
   - A chosen spectral filter -> an intrinsic manifold distance or kernel.
     Heat weights have the cleanest Brownian-time interpretation; inverse
     powers are Green's-function / fractional-Laplacian filters and can be
     viewed as scale mixtures of heat kernels, not as one Brownian time.
   - Entropic OT + Brownian/noisy interpolants + score/flow matching -> SB
     approximation ([SF]$^2$M-style).
   The building blocks exist in the literature but composing them into one
   formal convergence statement for our exact method is open work; we make
   the connection as motivation and support it empirically.

## Open questions worth flagging in "Future Work"

- Compose the three approximation arguments into a formal convergence
  statement.
- Add stochasticity (entropic OT + Brownian bridge + score) to recover the
  finite-$\varepsilon$ regime.
- Test whether the optimal $\alpha$ correlates with a measurable manifold
  property (e.g., effective dimension, Fiedler value of the union graph).
- Extend the connectivity-conditioned ablation to graph augmentation as a
  way to handle "near-disconnected" datasets without falling back to
  Euclidean OT.

## Files in this repo that anchor each connection

- `surf/ot/costs.py:compute_biharmonic_cost_matrix` — power-law spectral
  cost. The `weight_power` parameter implements $1/\lambda^{2 \cdot
  \text{weight\_power}}$, mapped from the user-facing $\alpha$ via
  `surf/training/method_registry.py:_PARAM_KEY_TO_KW`.
- `surf/training/method_registry.py:resolve_method_kwargs` — dispatches
  `MM+SLERP+SquaredSpectral@alpha=...` and `MM+Linear+SquaredSpectral@alpha=...`
  to `make_spectral_cost_fn` with the appropriate $\alpha$ binding.
- `scripts/connectivity_diagnostic.py` — produces
  `surf_latex/final_report/figures/connectivity_diagnostic.pdf`, the kNN
  union-graph component-count plot that empirically tells us when the
  spectral cost is well-defined.
- `surf/evaluation/metrics.py:mmd_otpfm` — multi-scale RBF MMD$^2$ matching
  the OTP-FM reference implementation; needed for direct comparison to
  OTP-FM Table 2 published numbers.
