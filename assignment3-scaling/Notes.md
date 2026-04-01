# Overview

**Goal**
Find the **compute-optimal model** (lowest training loss) under a fixed FLOPs budget of **1e19**.
- Larger model + less data **vs.** smaller model + more data.

**Approach**
Use **scaling laws** to empirically relate training loss to model size and compute.
- Kaplan et al. (2020)
- Hoffmann et al. (2022) — Chinchilla

**Setup**
- No direct training — call a **training API** instead
- **API inputs:** model hyperparameters (layers, embedding size, # heads, batch size, lr) + desired FLOPs
- **API output:** final training loss

**Budgets**
| Purpose | FLOPs |
|---|---|
| Scaling law exploration | 2e18 |
| Final big run target | 1e19 |

**Core Challenge**
Efficiently use the **2e18 exploration budget** to collect enough data points,
fit a reliable scaling law, then **extrapolate** to find the best config at 1e19 FLOPs.


# Scaling Law Review
## Scaling Laws from IsoFLOPs profiles
Given a fixed compute budget **C**, what model size **N** and data size **D**
give the **lowest training loss**?

**Key Formula**

$$C = 6ND$$

| Symbol | Meaning |
|--------|---------|
| C | Compute budget (FLOPs) |
| N | Number of model parameters |
| D | Number of training tokens |

**Trade-off:** C is fixed, so larger N → smaller D, and vice versa.

### IsoFLOPs Approach (Hoffmann et al., 2022)

1. Fix several compute budgets
Choose a set of budgets: C₁, C₂, C₃, ...

2. For each budget Cᵢ, train models of varying sizes Nᵢⱼ
   - Keep total FLOPs = Cᵢ for all runs
   - Set D = Cᵢ / (6N) automatically

3. Observe the U-shaped loss curve
For each budget Cᵢ, plotting loss vs. N yields a **quadratic (U-shaped)** curve:
    ```
    Loss
    ↑
    |  *                  *       ← N too small (underfitting)
    |      *          *           ← N too large (not enough data)
    |          *  *               ← sweet spot: N_opt(Cᵢ)
    └─────────────────────→ N
    ```
    - **N too small** → model too weak to fit data → high loss
    - **N too large** → D = C/6N becomes tiny, model undertrained → high loss
    - **N_opt** → best balance between model capacity and training data
4. Collect optimal points
Repeat for each budget → get a sequence of pairs:

$$\langle C_i,\ N_{\text{opt}}(C_i) \rangle$$

### Fitting Power Laws

Empirically, the optimal values follow **power laws**:
$$N_{\text{opt}} \propto C^a$$$$D_{\text{opt}} \propto C^b$$

In log space, these are straight lines → easy to fit via linear regression:

$$\log N_{\text{opt}} = a \cdot \log C + \text{const}$$

### Extrapolation to Target Budget

Once power laws are fitted from small-scale experiments,
plug in the **target budget** (1e19 FLOPs) to predict:

- **Optimal model size** → N_opt
- **Optimal training tokens** → D_opt

### Problem: chinchilla_isoflops


Reproduce the IsoFLOPs scaling law method from Hoffmann et al. (2022) using provided training run data. For each compute budget, identify the optimal model size and dataset size, fit power laws, and extrapolate to larger budgets.

- File: `isoflops_curves.json`
- 9 compute budgets: 6e18 → 3e21 FLOPs
- Each budget has ~8 runs with varying model sizes N
- Each run records: `parameters` (N), `compute_budget` (C), `final_loss` (L)

1. Find `N_opt` for each budget
For each compute budget Cᵢ, select the run with the **lowest final loss** (rather than fitting a quadratic, as simplified from the original paper):
    | C (FLOPs) | N_opt (params) | D_opt (tokens) | Loss |
    |-----------|----------------|----------------|------|
    | 6e18 | 7.621e8 | 1.312e9 | 5.8999 |
    | 1e19 | 8.066e8 | 2.066e9 | 5.6179 |
    | 3e19 | 1.537e9 | 3.253e9 | 5.1072 |
    | 6e19 | 1.952e9 | 5.123e9 | 4.8306 |
    | 1e20 | 3.253e9 | 5.123e9 | 4.6529 |
    | 3e20 | 5.904e9 | 8.469e9 | 4.3112 |
    | 6e20 | 6.971e9 | 1.435e10 | 4.1212 |
    | 1e21 | 6.859e9 | 2.430e10 | 4.0028 |
    | 3e21 | 1.215e10 | 4.116e10 | 3.7732 |

2. Derive `D_opt`
From the fundamental relation `C = 6ND`:
$$D_{\text{opt}} = \frac{C}{6 \times N_{\text{opt}}}$$

3. Fit power laws via `curve_fit` (log-space)
Model: `log(y) = a * log(C) + log(k)`, fit using `scipy.optimize.curve_fit`

$$N_{\text{opt}} = 1.163 \times C^{0.469}$$$$D_{\text{opt}} = 0.143 \times C^{0.531}$$
- **Sanity check:** exponents sum to ≈ 1.0, consistent with C = 6ND

#### Results
**Q1 – Optimal Model Size**

| Budget | Predicted N_opt |
|--------|----------------|
| 1e23 FLOPs | **~7.0 × 10¹⁰ (~70B parameters)** |
| 1e24 FLOPs | **~2.1 × 10¹¹ (~206B parameters)** |

**Q2 – Optimal Dataset Size**

| Budget | Predicted D_opt |
|--------|----------------|
| 1e23 FLOPs | **~2.4 × 10¹¹ (~240B tokens)** |
| 1e24 FLOPs | **~8.1 × 10¹¹ (~810B tokens)** |

**Key Implementation Details**
- **N_opt selection:** argmin of final_loss per budget (no quadratic fitting)
- **Power law fitting:** `scipy.optimize.curve_fit` on log-transformed values
- **Extrapolation:** plug target C into fitted power law formula
- **Plot:** log-log scale, data points + fitted curve + predictions at 1e23, 1e24

**Takeaway**
As compute budget grows, both model size and dataset size should scale roughly as C^0.5 — meaning compute should be split approximately equally between making the model larger and training on more data.
$C\propto N_{opt} \cdot D_{opt} \propto C^{0.469} \cdot C^{0.531} = C^{1.0}$
This aligns with the Chinchilla finding that most prior models were significantly undertrained.

# Constructing Scaling Laws
## Experimental Design Report

**Goal:** Use a budget of `≤ 2e18 FLOPs` to fit a scaling law, then predict the compute-optimal model size, hyperparameters, and training loss for `C = 1e19` FLOPs.

**Key constraint:** Batch size must be 128 or 256.

**Non-embedding parameter formula:**
$$N = 12 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2$$

### Overall Strategy

We follow the **IsoFLOPs approach** (Hoffmann et al., 2022):

1. Choose several small compute budgets C₁, C₂, ...
2. For each budget, train models of varying sizes N (adjusting D = C/6N accordingly)
3. Find N_opt per budget (lowest loss)
4. Fit power laws: N_opt ∝ Cᵃ, D_opt ∝ Cᵇ
5. Extrapolate to C = 1e19

### Budget Allocation Plan

Total budget: **2e18 FLOPs**

We split the budget into two phases:

| Phase | Purpose | Budget |
|-------|---------|--------|
| Phase 1 | Hyperparameter tuning (lr sweep) | 2e17 |
| Phase 2 | IsoFLOPs profiling (main scaling law) | 1.8e18 |

**Breakdown**

We choose **3 compute budgets**, each with **6 model sizes**:

| IsoFLOP Budget | # Models | Subtotal |
|---------------|----------|----------|
| C₁ = 6e16 | 6 | 3.6e17 |
| C₂ = 1.5e17 | 6 | 9.0e17 |
| C₃ = 3e17 | 4 | 1.2e17 |
| Phase 1 (lr sweep) | 5 | ~1.5e17 |
| **Total** | | **≈ 1.77e18 ✅** |



### Model Size Candidates

Using N = 12 · n_layers · d_model², we pre-compute candidate architectures.
We need models spanning roughly **1 order of magnitude** in N for each profile.

| n_layers | d_model | n_heads | N (params) |
|----------|---------|---------|------------|
| 4 | 256 | 4 | ~786K |
| 6 | 384 | 6 | ~2.65M |
| 8 | 512 | 8 | ~6.29M |
| 12 | 512 | 8 | ~9.44M |
| 12 | 768 | 12 | ~21.2M |
| 16 | 768 | 12 | ~28.3M |
| 16 | 1024 | 16 | ~50.3M |
| 24 | 1024 | 16 | ~75.5M |
| 24 | 1280 | 16 | ~117.9M |

**Rules for valid architectures:**
- `d_model` must be divisible by `n_heads`
- `n_heads` typically = d_model / 64
- Larger models assigned to larger C budgets to ensure D = C/6N ≥ ~100 tokens/param

### Model Assignment to IsoFLOP Profiles

For each budget Cᵢ, we choose models such that D = Cᵢ / (6N) is reasonable (not too few tokens — at minimum ~10× the number of parameters):

**C₁ = 6e16:**
Models with N ~ 0.5M – 10M (small models, enough tokens to train)

**C₂ = 1.5e17:**
Models with N ~ 1M – 25M

**C₃ = 3e17:**
Models with N ~ 5M – 50M (fewer runs due to budget)

### Hyperparameter Tuning (Phase 1)

Before the main IsoFLOPs sweep, we tune the **learning rate** using small cheap runs.

**Learning Rate Sweep Design**

- Fix: C = 3e16, model = 6 layers / d_model=384 (~2.65M params)
- Vary lr ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
- Cost: 5 × 3e16 = 1.5e17 FLOPs

| lr | Expected behavior |
|----|------------------|
| 1e-4 | Possibly too slow, higher loss |
| 3e-4 | Standard good default |
| 1e-3 | Often optimal for small models |
| 3e-3 | May become unstable |
| 1e-2 | Likely diverges |

**Decision rule:** Pick the lr with lowest final loss. Fix this lr for all
subsequent Phase 2 runs.

**Other Fixed Hyperparameters**

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| Batch size | **256** | Allowed value; better GPU utilization |
| Optimizer | AdamW | Standard for LLMs |
| LR schedule | Cosine decay | Standard practice |
| Warmup steps | ~1% of total steps | Prevents early instability |
| Weight decay | 0.1 | Standard |
| Gradient clip | 1.0 | Stability |

### Fitting the Scaling Law

1. Identify N_opt per budget
For each Cᵢ, select the run with **minimum final loss** → gives N_opt(Cᵢ).
Derive D_opt = Cᵢ / (6 × N_opt).

2. Fit power laws
In log-space, fit linear regression using `scipy.optimize.curve_fit`:
$$\log N_{\text{opt}} = a \cdot \log C + \log k_N$$$$\log D_{\text{opt}} = b \cdot \log C + \log k_D$$

   - **Expected result:** a ≈ 0.5, b ≈ 0.5 (consistent with Chinchilla findings).
   - Sanity check: a + b ≈ 1.0 (required by C = 6ND).

3. Extrapolate to C = 1e19

$$N_{\text{opt}}(10^{19}) = k_N \cdot (10^{19})^a$$$$D_{\text{opt}}(10^{19}) = k_D \cdot (10^{19})^b$$

4. Predict training loss
After identifying N_opt, find or interpolate the predicted loss using a loss-vs-compute curve, or directly from the closest experimental data point.

**Mapping N_opt to Concrete Hyperparameters**

Once we have the predicted N_opt for C = 1e19, we find the architecture
(n_layers, d_model, n_heads) such that:

$$12 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2 \approx N_{\text{opt}}$$

**Strategy:**
- Fix aspect ratio: keep d_model / n_layers roughly constant (~consistent
  with standard LLM designs, e.g. d_model ≈ 128 × n_layers^0.5)
- Ensure n_heads = d_model / 64
- Round to nearest clean values

**Example:** If N_opt ≈ 100M:
- n_layers = 12, d_model = 768 → N = 12 × 12 × 768² ≈ 84.9M ✓
- n_layers = 16, d_model = 768 → N = 12 × 16 × 768² ≈ 113.2M ✓

Pick whichever is closest to predicted N_opt.



### Evaluation Criteria

| Question | How we answer it |
|----------|-----------------|
| Which runs to query? | IsoFLOPs profiles at 3 budgets + lr sweep |
| How to fit scaling law? | Power law via curve_fit in log-space |
| How well does it fit? | R² score on log-log plot; check a+b≈1 |
| Predicted N_opt at 1e19? | Extrapolate power law |
| Hyperparameters? | Map N_opt → (n_layers, d_model, n_heads) |

### Risk Management

| Risk | Mitigation |
|------|-----------|
| U-curve not visible (all losses monotone) | Extend model size range further |
| lr sweep uses too much budget | Reduce to 3 lr values |
| Power law fit has high error | Add one more IsoFLOP budget if budget allows |
| N_opt doesn't map cleanly to architecture | Use two nearest architectures, pick lower loss |

### Summary of Planned Runs

| Run Type | # Runs | FLOPs each | Total |
|----------|--------|-----------|-------|
| lr sweep | 5 | 3e16 | 1.5e17 |
| IsoFLOPs C₁=6e16 | 6 | 6e16 | 3.6e17 |
| IsoFLOPs C₂=1.5e17 | 6 | 1.5e17 | 9.0e17 |
| IsoFLOPs C₃=3e17 | 4 | 3e17 | 1.2e17 |
| **Total** | **21** | | **≈ 1.77e18 ✅** |

Buffer remaining: ~2.3e17 FLOPs for debugging or extra runs.

## Training Run Details

### Model Architecture
| Component | This Assignment |
|-----------|----------------|
| Positional encoding | Absolute embeddings |
| Normalization | Layer Norm |
| FFN | Linear → GeLU → Linear (hidden dim = 4×d_model) |
| Embeddings | Untied input/output |

### Training Setup
| Setting | Value |
|---------|-------|
| Dataset | SlimPajama |
| Tokenizer | Byte-level BPE, 32K vocab |
| Context length | 512 |
| Dropout | 0.1 (attention + residual) |
| Optimizer | AdamW, weight decay=0.01 |
| Gradient clipping | 1.0 |
| LR schedule | Cosine decay, 10× reduction, no warmup |

### Key Notes
- Parameter formula **still holds**: N = 12 · n_layers · d_model²
  (Attention: 4d² + FFN: 8d² = 12d² per layer ✅)
- API constrains lr ∈ [1e-4, 1e-3] — narrow range, choose carefully
- No LR warmup → large lr values may cause instability
