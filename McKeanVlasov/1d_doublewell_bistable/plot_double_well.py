"""
plot_double_well.py
-------------------
Loads checkpoints from train_double_well.py and produces all figures.

Figure layout
-------------
Fig 1 — 2x2 panel: initial conditions (top) and learned steady states (bottom)
Fig 2 — loss convergence for both networks  (aspect ratio height:width = 1:3)
Fig 3 — effective potential for both steady states
Fig 4 — Q-Q plots for both networks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

from exact_double_well import gibbs_density, self_consistent_means

# ============================================================================
# Reproduction of network architecture (must match train_double_well.py)
# ============================================================================

class PushforwardNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, x0, r):
        tr = torch.cat([t, r], dim=1)
        return x0 + torch.sqrt(t) * self.net(tr)


# ============================================================================
# Parameters (must match train_double_well.py)
# ============================================================================

KAPPA  = 0.5
SIGMA2 = 0.4
MU0    = 0.8
SIGMA0 = 0.3
T      = 1.0

RESULTS_DIR = 'results_double_well'
FIGS_DIR    = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Load checkpoints
# ============================================================================

def load_checkpoint(side):
    fname = os.path.join(RESULTS_DIR, f'checkpoint_{side}.pt')
    ckpt  = torch.load(fname, map_location=device)
    net   = PushforwardNet(hidden=32).to(device)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    return net, ckpt


def draw_samples(net, mu0_bias, n=50000):
    """Draw samples from rho(T, .) using the trained pushforward."""
    with torch.no_grad():
        x0 = mu0_bias + SIGMA0 * torch.randn(n, 1, device=device)
        r  = torch.randn(n, 1, device=device)
        t  = T * torch.ones(n, 1, device=device)
        xs = net(t, x0, r).cpu().numpy().flatten()
    return xs


def draw_ic_samples(mu0_bias, n=50000):
    """Draw samples from the initial condition N(mu0_bias, sigma0^2)."""
    return np.random.normal(mu0_bias, SIGMA0, n)


# ============================================================================
# Exact steady states
# ============================================================================

def get_exact_steady_states():
    """Return (xs, rho) for all fixed points, labelled by stability."""
    fixed_pts = self_consistent_means(KAPPA, SIGMA2)
    xs_grid   = np.linspace(-3.5, 3.5, 1000)
    results   = []
    for m_star, stab in fixed_pts:
        _, rho = gibbs_density(m_star, KAPPA, SIGMA2, xs_grid)
        results.append({'m': m_star, 'stability': stab,
                        'xs': xs_grid, 'rho': rho})
    return results


# ============================================================================
# Helper: histogram + density overlay
# ============================================================================

def plot_hist_and_density(ax, samples, xs, rho, color, label_hist, label_density,
                          title, show_density=True):
    ax.hist(samples, bins=80, density=True, color=color,
            alpha=0.45, label=label_hist)
    if show_density:
        ax.plot(xs, rho, color=color, lw=2.0, label=label_density)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('density', fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(-3.5, 3.5)
    ax.grid(True, alpha=0.25)


# ============================================================================
# Figure 1 — 2x2 main panel
# ============================================================================

def fig_main_panel(net_left, ckpt_left, net_right, ckpt_right, exact_states):
    """
    Upper row : initial conditions (histogram only)
    Lower row : learned steady states vs exact Gibbs density
    """
    # Identify which exact state is left (m<0) and right (m>0)
    stable_states = [s for s in exact_states if s['stability'] == 'stable']
    state_neg = next(s for s in stable_states if s['m'] < 0)
    state_pos = next(s for s in stable_states if s['m'] > 0)

    # Samples
    ic_left   = draw_ic_samples(-MU0)
    ic_right  = draw_ic_samples(+MU0)
    ss_left   = draw_samples(net_left,  -MU0)
    ss_right  = draw_samples(net_right, +MU0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(
        rf'1D Double-Well McKean–Vlasov  '
        rf'($\kappa={KAPPA}$, $\sigma^2={SIGMA2}$)',
        fontsize=13
    )

    col_left  = '#2166ac'   # blue
    col_right = '#d6604d'   # red

    # ---- upper left: initial condition biased left ----
    xs_ic = np.linspace(-3.5, 3.5, 500)
    rho_ic_left  = (1.0 / (SIGMA0 * np.sqrt(2*np.pi)) *
                    np.exp(-0.5 * ((xs_ic + MU0) / SIGMA0)**2))
    plot_hist_and_density(
        axes[0, 0], ic_left,
        xs_ic, rho_ic_left,
        col_left,
        label_hist=r'samples from $\rho_0$',
        label_density=r'$\rho_0 = \mathcal{N}(-0.8,\,0.09)$',
        title=r'Initial condition (left-biased)',
    )

    # ---- upper right: initial condition biased right ----
    rho_ic_right = (1.0 / (SIGMA0 * np.sqrt(2*np.pi)) *
                    np.exp(-0.5 * ((xs_ic - MU0) / SIGMA0)**2))
    plot_hist_and_density(
        axes[0, 1], ic_right,
        xs_ic, rho_ic_right,
        col_right,
        label_hist=r'samples from $\rho_0$',
        label_density=r'$\rho_0 = \mathcal{N}(+0.8,\,0.09)$',
        title=r'Initial condition (right-biased)',
    )

    # ---- lower left: learned steady state (left), m* < 0 ----
    m_left = ckpt_left['m_star']
    plot_hist_and_density(
        axes[1, 0], ss_left,
        state_neg['xs'], state_neg['rho'],
        col_left,
        label_hist=rf'learned $\rho^*(x)$  ($\hat{{m}}^*={m_left:.3f}$)',
        label_density=rf'exact Gibbs ($m^*={state_neg["m"]:.3f}$)',
        title=r'Steady state (left-biased network)',
    )

    # ---- lower right: learned steady state (right), m* > 0 ----
    m_right = ckpt_right['m_star']
    plot_hist_and_density(
        axes[1, 1], ss_right,
        state_pos['xs'], state_pos['rho'],
        col_right,
        label_hist=rf'learned $\rho^*(x)$  ($\hat{{m}}^*={m_right:.3f}$)',
        label_density=rf'exact Gibbs ($m^*={state_pos["m"]:.3f}$)',
        title=r'Steady state (right-biased network)',
    )

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig1_main_panel.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Figure 2 — loss convergence  (height:width = 1:3)
# ============================================================================

def fig_loss(ckpt_left, ckpt_right):
    """
    Two-panel loss figure (height:width = 1:3 each panel).
    Left panel  : Stage 1 (epsilon decay)
    Right panel : Stage 2 (stationary)
    A vertical dashed line marks the stage boundary on a combined x-axis
    version shown as an inset title.
    """
    loss1_l = np.array(ckpt_left['loss_log_stage1'])
    loss2_l = np.array(ckpt_left['loss_log_stage2'])
    loss1_r = np.array(ckpt_right['loss_log_stage1'])
    loss2_r = np.array(ckpt_right['loss_log_stage2'])

    n1 = len(loss1_l)
    n2 = len(loss2_l)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))   # total width:height ~ 3:1

    col_l = '#2166ac'
    col_r = '#d6604d'

    # ---- Stage 1 ----
    ep1 = np.arange(1, n1 + 1)
    axes[0].semilogy(ep1, loss1_l, color=col_l, lw=1.2, label='left-biased')
    axes[0].semilogy(ep1, loss1_r, color=col_r, lw=1.2, label='right-biased')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Stage 1 — epsilon decay', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # ---- Stage 2 ----
    ep2 = np.arange(1, n2 + 1)
    axes[1].semilogy(ep2, loss2_l, color=col_l, lw=1.2, label='left-biased')
    axes[1].semilogy(ep2, loss2_r, color=col_r, lw=1.2, label='right-biased')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Stage 2 — stationary', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig2_loss.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Figure 3 — effective potential
# ============================================================================

def fig_effective_potential(exact_states):
    xs = np.linspace(-2.5, 2.5, 500)

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {'stable_neg': '#2166ac', 'stable_pos': '#d6604d', 'unstable': '#888888'}

    for s in exact_states:
        phi = xs**4 / 4.0 + (KAPPA - 1.0) / 2.0 * xs**2 - KAPPA * s['m'] * xs
        if s['stability'] == 'unstable':
            ls, c, lbl = '--', colors['unstable'], rf"$m^*=0$ (unstable)"
        elif s['m'] < 0:
            ls, c, lbl = '-',  colors['stable_neg'], rf"$m^*={s['m']:.3f}$ (stable)"
        else:
            ls, c, lbl = '-',  colors['stable_pos'], rf"$m^*={s['m']:.3f}$ (stable)"
        ax.plot(xs, phi, ls=ls, color=c, lw=2.0, label=lbl)

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel(r'$\Phi(x;\,m^*)$', fontsize=11)
    ax.set_title('Effective potential for each steady state', fontsize=12)
    ax.set_ylim(-0.8, 1.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig3_potential.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Figure 4 — Q-Q plots
# ============================================================================

def fig_qq(net_left, net_right, exact_states):
    stable_states = [s for s in exact_states if s['stability'] == 'stable']
    state_neg = next(s for s in stable_states if s['m'] < 0)
    state_pos = next(s for s in stable_states if s['m'] > 0)

    def exact_quantiles(state, probs):
        """Compute quantiles of the exact Gibbs density by CDF inversion."""
        xs, rho = state['xs'], state['rho']
        dx  = xs[1] - xs[0]
        cdf = np.cumsum(rho) * dx
        cdf = np.clip(cdf, 0.0, 1.0)
        return np.interp(probs, cdf, xs)

    probs = np.linspace(0.01, 0.99, 200)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle('Q–Q plots: learned vs exact Gibbs', fontsize=12)

    for ax, net, state, bias, color, title in [
        (axes[0], net_left,  state_neg, -MU0, '#2166ac', 'Left-biased network'),
        (axes[1], net_right, state_pos, +MU0, '#d6604d', 'Right-biased network'),
    ]:
        samples  = draw_samples(net, bias, n=20000)
        q_exact  = exact_quantiles(state, probs)
        q_learned = np.quantile(samples, probs)

        ax.plot(q_exact, q_learned, '.', color=color, ms=3.0, alpha=0.7)
        lim = (min(q_exact.min(), q_learned.min()) - 0.1,
               max(q_exact.max(), q_learned.max()) + 0.1)
        ax.plot(lim, lim, 'k--', lw=1.0, label='diagonal')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('Exact quantile', fontsize=10)
        ax.set_ylabel('Learned quantile', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig4_qq.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Loading checkpoints...")
    net_left,  ckpt_left  = load_checkpoint('left')
    net_right, ckpt_right = load_checkpoint('right')

    print("Computing exact steady states...")
    exact_states = get_exact_steady_states()
    for s in exact_states:
        print(f"  m* = {s['m']:+.4f}  ({s['stability']})")

    print("Generating Figure 1 (main 2x2 panel)...")
    fig_main_panel(net_left, ckpt_left, net_right, ckpt_right, exact_states)

    print("Generating Figure 2 (loss convergence)...")
    fig_loss(ckpt_left, ckpt_right)

    print("Generating Figure 3 (effective potential)...")
    fig_effective_potential(exact_states)

    print("Generating Figure 4 (Q-Q plots)...")
    fig_qq(net_left, net_right, exact_states)

    print("\nAll figures saved to", FIGS_DIR)