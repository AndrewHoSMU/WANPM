import matplotlib; matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PARAMETERS — must match fFP_20d_doublewell.py
# ============================================================================
ALPHA   = 1.5
DIM     = 20
V_ALPHA = 1.0
T_FINAL = 1.0
IC_SIGMA = 0.5
SNAPSHOT_TIMES = [0.0, 0.1, 0.2, 0.5, 0.8, 1.0]

N_PARTICLES = 20_000
N_STEPS     = 500          # h = T_FINAL / N_STEPS = 0.002


# ============================================================================
# DRIFT  b(x) = -grad V,  V(x) = V_ALPHA * ||x - c+||^2 * ||x - c-||^2
# c+ = +1,  c- = -1
# ============================================================================
def drift(x):
    """x: (N, D) -> b: (N, D)"""
    c_plus  = np.ones(DIM)
    c_minus = -np.ones(DIM)
    diff_p  = x - c_plus               # (N, D)
    diff_m  = x - c_minus              # (N, D)
    norm_p2 = (diff_p ** 2).sum(axis=1, keepdims=True)   # (N, 1)
    norm_m2 = (diff_m ** 2).sum(axis=1, keepdims=True)   # (N, 1)
    b = -2.0 * V_ALPHA * (diff_p * norm_m2 + diff_m * norm_p2)
    return b


# ============================================================================
# ISOTROPIC α-STABLE INCREMENT  (sub-Gaussian representation)
#
# For a positive stable subordinator A with E[e^{-sA}] = e^{-s^γ}, γ = α/2,
# we use Devroye's exact formula (works for γ ∈ (0,1)):
#
#   V ~ Uniform(0, π),  W ~ Exponential(1)
#   A = sin(γV)^{1/γ} * sin((1-γ)V)^{(1-γ)/γ} / (sin(V) * W^{(1-γ)/γ})
#
# Then the isotropic α-stable increment over time step h is:
#
#   ΔL = h^{1/α} * sqrt(2A) * G,   G ~ N(0, I_d)
#
# Characteristic function check:
#   E[e^{iξ·ΔL}] = E[e^{-|ξ|²·h^{2/α}·A}]
#                = e^{-(|ξ|²·h^{2/α})^{α/2}}
#                = e^{-h·|ξ|^α}   ✓   (matches generator of α-stable Lévy)
# ============================================================================
def sample_stable_increments(n, d, alpha, h):
    gamma = alpha / 2.0                        # index of positive stable
    V = np.random.uniform(0.0, np.pi, size=(n, 1))
    W = np.random.exponential(1.0,    size=(n, 1))
    # Positive stable subordinator A, E[e^{-sA}] = e^{-s^gamma}
    A = (np.sin(gamma * V) ** (1.0 / gamma)
         * np.sin((1.0 - gamma) * V) ** ((1.0 - gamma) / gamma)
         / (np.sin(V) * W ** ((1.0 - gamma) / gamma)))
    G = np.random.randn(n, d)
    return h ** (1.0 / alpha) * np.sqrt(2.0 * A) * G


# ============================================================================
# EULER–MARUYAMA SIMULATION
# ============================================================================
def simulate():
    h = T_FINAL / N_STEPS
    print(f"  N_PARTICLES={N_PARTICLES}, N_STEPS={N_STEPS}, h={h:.4f}")

    X = IC_SIGMA * np.random.randn(N_PARTICLES, DIM)

    snapshots = {}
    if 0.0 in SNAPSHOT_TIMES:
        snapshots[0.0] = X.copy()

    t = 0.0
    for step in range(N_STEPS):
        b     = drift(X)
        noise = sample_stable_increments(N_PARTICLES, DIM, ALPHA, h)
        X     = X + h * b + noise
        t    += h

        for t_snap in SNAPSHOT_TIMES:
            if t_snap > 0.0 and t_snap not in snapshots and abs(t - t_snap) < h * 0.5 + 1e-12:
                snapshots[t_snap] = X.copy()
                print(f"  Snapshot saved: t={t_snap}")

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{N_STEPS}  t={t:.3f}")

    return snapshots


# ============================================================================
# PLOT  — same layout as 20d_doublewell_projections.png
# ============================================================================
def plot_results(snapshots):
    proj_pairs = [(0, 1), (0, DIM - 1)]
    n_snaps = len(SNAPSHOT_TIMES)
    n_proj  = len(proj_pairs)

    fig, axes = plt.subplots(n_proj, n_snaps, figsize=(4 * n_snaps, 4 * n_proj))

    for j, t_val in enumerate(SNAPSHOT_TIMES):
        x_samples = snapshots[t_val]
        for i, (d1, d2) in enumerate(proj_pairs):
            ax = axes[i, j]
            ax.scatter(x_samples[:, d1], x_samples[:, d2],
                       s=1, alpha=0.3, color='firebrick', rasterized=True)
            ax.set_title(f't={t_val}\n(x{d1+1},x{d2+1})', fontsize=9)
            ax.set_xlabel(f'x{d1+1}')
            ax.set_ylabel(f'x{d2+1}')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Fractional 20D Double-Well  (α={ALPHA})  —  Particle Method (Euler–Maruyama)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, '20d_doublewell_projections_particle.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {out_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print(f"Particle verification  |  α={ALPHA}  dim={DIM}  T={T_FINAL}")
    print("=" * 70)
    snapshots = simulate()
    plot_results(snapshots)
    print("Done.")


if __name__ == '__main__':
    main()
