import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ── 1. Load data ──────────────────────────────────────────────────────────────
with open("/mnt/user-data/uploads/isoflops_curves.json") as f:
    runs = json.load(f)

# ── 2. Find N_opt and D_opt for each compute budget ───────────────────────────
from collections import defaultdict

budget_runs = defaultdict(list)
for r in runs:
    budget_runs[r["compute_budget"]].append(r)

budgets, N_opts, D_opts = [], [], []
for C, group in sorted(budget_runs.items()):
    best = min(group, key=lambda x: x["final_loss"])
    N = best["parameters"]
    D = C / (6 * N)   # from C = 6ND
    budgets.append(C)
    N_opts.append(N)
    D_opts.append(D)
    print(f"C={C:.1e}  →  N_opt={N:.3e},  D_opt={D:.3e},  loss={best['final_loss']:.4f}")

budgets  = np.array(budgets,  dtype=float)
N_opts   = np.array(N_opts,   dtype=float)
D_opts   = np.array(D_opts,   dtype=float)

# ── 3. Fit power laws in log space: log(N) = a*log(C) + log(k) ───────────────
def power_law(log_C, a, log_k):
    return a * log_C + log_k

pN, _ = curve_fit(power_law, np.log(budgets), np.log(N_opts))
pD, _ = curve_fit(power_law, np.log(budgets), np.log(D_opts))

aN, kN = pN[0], np.exp(pN[1])
aD, kD = pD[0], np.exp(pD[1])

print(f"\nN_opt power law:  N = {kN:.4e} * C^{aN:.4f}")
print(f"D_opt power law:  D = {kD:.4e} * C^{aD:.4f}")

# ── 4. Extrapolate ────────────────────────────────────────────────────────────
for C_target in [1e23, 1e24]:
    N_pred = kN * C_target**aN
    D_pred = kD * C_target**aD
    print(f"\nC = {C_target:.0e}")
    print(f"  Predicted N_opt = {N_pred:.4e} parameters")
    print(f"  Predicted D_opt = {D_pred:.4e} tokens")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
C_extrap = np.logspace(np.log10(budgets.min()), 24, 300)
N_fit    = kN * C_extrap**aN
D_fit    = kD * C_extrap**aD

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("IsoFLOPs Scaling Laws", fontsize=14, fontweight="bold")

COLORS = {"data": "#2196F3", "fit": "#F44336", "pred": "#4CAF50"}

for ax, y_data, y_fit, label, unit in zip(
    axes,
    [N_opts, D_opts],
    [N_fit,  D_fit],
    ["N_opt (Model Parameters)", "D_opt (Training Tokens)"],
    ["Parameters", "Tokens"],
):
    ax.scatter(budgets, y_data, color=COLORS["data"], zorder=5,
               s=80, label="Observed optimal points")
    ax.plot(C_extrap, y_fit, color=COLORS["fit"], linewidth=2,
            label=f"Power-law fit (exponent={aN:.3f})" if "Parameters" in unit
                  else f"Power-law fit (exponent={aD:.3f})")

    # Mark predictions at 1e23 and 1e24
    for C_t, marker in [(1e23, "^"), (1e24, "s")]:
        y_pred = kN * C_t**aN if "Parameters" in unit else kD * C_t**aD
        ax.scatter(C_t, y_pred, color=COLORS["pred"], zorder=6,
                   s=120, marker=marker,
                   label=f"C=1e{int(np.log10(C_t))} → {y_pred:.2e} {unit}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget C (FLOPs)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/isoflops_scaling_laws.png", dpi=150, bbox_inches="tight")
print("\nPlot saved.")
