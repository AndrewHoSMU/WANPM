"""
train_double_well.py
--------------------
Trains two pushforward networks for the 1D double-well McKean-Vlasov equation
using a two-stage training strategy.

Stage 1 — epsilon decay  (epochs 0 .. N_EPOCHS_STAGE1-1):
    Solves the epsilon-scaled time-dependent PDE
        eps * d_t rho = L[rho]
    with eps decaying linearly from EPS_START to 0 over this stage.
    The sqrt(t) architecture enforces rho(0,.) = rho_0 exactly, so the
    initial condition bias selects the correct basin throughout.

Stage 2 — stationary  (epochs 0 .. N_EPOCHS_STAGE2-1):
    Fixes t=1 and minimises the pure stationary weak-form residual
        E_{rho}[ b(x,m)*df/dx + sigma^2/2 * d^2f/dx^2 ] = 0
    with no time-derivative terms.  Network is warm-started from Stage 1,
    already sitting in the correct basin.

PDE drift:
    b(x, m) = -V'(x) - kappa*(x - m) = -x^3 + (1-kappa)*x + kappa*m
    V(x) = x^4/4 - x^2/2

Two networks are trained:
  - 'left':  initial condition N(-MU0, SIGMA0^2)  =>  targets m* < 0
  - 'right': initial condition N(+MU0, SIGMA0^2)  =>  targets m* > 0

Outputs saved to results_double_well/:
  checkpoint_left.pt
  checkpoint_right.pt
"""

import os
import time as pytime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# Problem parameters
# ============================================================================

KAPPA  = 0.5          # Interaction strength  (< 1 for bistability)
SIGMA2 = 0.4          # sigma^2
SIGMA  = SIGMA2**0.5  # sigma

T      = 1.0          # Time horizon
T_EPS  = 1e-3         # Small t offset to avoid sqrt(t) singularity at t=0

# Initial condition bias: N(+/- MU0, SIGMA0^2)
MU0    = 0.8
SIGMA0 = 0.3

# ============================================================================
# Training hyperparameters
# ============================================================================

K          = 2000   # Number of test functions
M          = 2000   # Batch size (both stages)
M0         = 500    # Initial-condition batch size (Stage 1 only)
MT         = 500    # Terminal batch size (Stage 1 only)
LR_GEN     = 1e-2   # Generator learning rate (Adam)
LR_TEST    = 1e-2   # Adversary learning rate (SGD)
FREQ_SCALE = 2.0    # Test-function frequency initialisation scale

# Stage 1: epsilon-decay
N_EPOCHS_STAGE1 = 25000
EPS_START       = 1.0   # epsilon at epoch 0 of Stage 1
                        # decays linearly to 0 by end of Stage 1

# Stage 2: stationary fine-tuning
N_EPOCHS_STAGE2 = 25000

# ============================================================================
# Device
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

os.makedirs('results_double_well', exist_ok=True)


# ============================================================================
# Networks
# ============================================================================

class PushforwardNet(nn.Module):
    """
    F_theta(t, x0, r) = x0 + sqrt(t) * net(t, r)

    Enforces rho(0,.) = rho_0 exactly for all epsilon values.
    Architecture: (2, 32, 32, 1) with Tanh activations.
    Input to net: [t, r]  (both scalars in 1D)
    """

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, x0, r):
        """
        t  : (B, 1)
        x0 : (B, 1)
        r  : (B, 1)
        returns x : (B, 1)
        """
        tr = torch.cat([t, r], dim=1)   # (B, 2)
        return x0 + torch.sqrt(t) * self.net(tr)


class TestFunctions(nn.Module):
    """
    f^(k)(x, t) = sin(w^(k)*x + kappa_t^(k)*t + b^(k))

    Frequencies w initialised at FREQ_SCALE to avoid spurious minimisers.
    kappa_t is used in Stage 1 (time-dependent); ignored in Stage 2.
    """

    def __init__(self, K, freq_scale=FREQ_SCALE):
        super().__init__()
        self.w       = nn.Parameter(torch.randn(1, K) * freq_scale)
        self.kappa_t = nn.Parameter(torch.randn(1, K))
        self.b       = nn.Parameter(torch.rand(1, K) * 2 * np.pi)

    def _phase(self, x, t):
        return x * self.w + t * self.kappa_t + self.b   # (B, K)

    def f(self, x, t):
        return torch.sin(self._phase(x, t))              # (B, K)

    def df_dt(self, x, t):
        return self.kappa_t * torch.cos(self._phase(x, t))   # (B, K)

    def df_dx(self, x, t):
        return self.w * torch.cos(self._phase(x, t))         # (B, K)

    def d2f_dx2(self, x, t):
        return -(self.w**2) * torch.sin(self._phase(x, t))   # (B, K)


# ============================================================================
# Loss functions
# ============================================================================

def sample_x0(mu0_bias, n):
    return mu0_bias + SIGMA0 * torch.randn(n, 1, device=device)


def drift(x, m_hat):
    """b(x, m) = -x^3 + (1-kappa)*x + kappa*m"""
    return -x**3 + (1.0 - KAPPA) * x + KAPPA * m_hat


def loss_stage1(net, tf, eps, mu0_bias):
    """
    Stage 1 loss: weak form of  eps * d_t rho = L[rho]  over (0, T).

    Integration by parts in both x and t gives three terms:

      Interior : (T - T_EPS) * E[ eps * d_t f  +  b(x,m)*df/dx  +  sigma^2/2 * d^2f/dx^2 ]
      Terminal :  eps * E_{rho(T)}[ f(T) ]
      Initial  : -eps * E_{rho_0}[ f(0) ]

    Residual = Terminal - Initial - Interior,  loss = mean(residual^2).
    """

    # ---- interior --------------------------------------------------------
    t_int  = T_EPS + (T - T_EPS) * torch.rand(M, 1, device=device)
    x0_int = sample_x0(mu0_bias, M)
    r_int  = torch.randn(M, 1, device=device)

    x_int  = net(t_int, x0_int, r_int)          # (M, 1)
    m_hat  = x_int.mean()                        # gradient flows through

    Lf_int = (drift(x_int, m_hat) * tf.df_dx(x_int, t_int)
              + (SIGMA2 / 2.0) * tf.d2f_dx2(x_int, t_int))   # (M, K)

    E_int  = (T - T_EPS) * (eps * tf.df_dt(x_int, t_int) + Lf_int).mean(dim=0)  # (K,)

    # ---- terminal --------------------------------------------------------
    x0_T  = sample_x0(mu0_bias, MT)
    r_T   = torch.randn(MT, 1, device=device)
    t_T   = T * torch.ones(MT, 1, device=device)
    x_T   = net(t_T, x0_T, r_T)
    E_T   = eps * tf.f(x_T, t_T).mean(dim=0)                 # (K,)

    # ---- initial ---------------------------------------------------------
    x0_ic = sample_x0(mu0_bias, M0)
    t_0   = torch.zeros(M0, 1, device=device)
    E_0   = eps * tf.f(x0_ic, t_0).mean(dim=0)               # (K,)

    # ---- residual --------------------------------------------------------
    residual = E_T - E_0 - E_int                              # (K,)
    return (residual**2).mean()


def loss_stage2(net, tf, mu0_bias):
    """
    Stage 2 loss: pure stationary weak form at t = 1.

        E_{rho*}[ b(x,m)*df/dx + sigma^2/2 * d^2f/dx^2 ] = 0

    No time-derivative terms; t is fixed to T=1.
    """
    t_one  = T * torch.ones(M, 1, device=device)
    x0     = sample_x0(mu0_bias, M)
    r      = torch.randn(M, 1, device=device)

    x      = net(t_one, x0, r)                               # (M, 1)
    m_hat  = x.mean()                                        # gradient flows through

    Lf     = (drift(x, m_hat) * tf.df_dx(x, t_one)
              + (SIGMA2 / 2.0) * tf.d2f_dx2(x, t_one))      # (M, K)

    residual = Lf.mean(dim=0)                                # (K,)
    return (residual**2).mean()


# ============================================================================
# Training loop for one network
# ============================================================================

def train_one(side):
    """
    Train a single network through both stages.

    Parameters
    ----------
    side : 'left' or 'right'

    Returns
    -------
    net               : trained PushforwardNet
    loss_log_stage1   : list of float
    loss_log_stage2   : list of float
    """
    mu0_bias = -MU0 if side == 'left' else MU0
    print(f"\n{'='*60}")
    print(f"Training {side} network  (rho_0 ~ N({mu0_bias:+.1f}, {SIGMA0**2:.2f}))")
    print(f"{'='*60}")

    net = PushforwardNet(hidden=32).to(device)
    tf  = TestFunctions(K).to(device)

    gen_opt  = optim.Adam(net.parameters(), lr=LR_GEN)
    test_opt = optim.SGD(tf.parameters(),   lr=LR_TEST)

    # ------------------------------------------------------------------
    # Stage 1: epsilon decay
    # ------------------------------------------------------------------
    print(f"\n  --- Stage 1: epsilon decay ({N_EPOCHS_STAGE1} epochs) ---")
    loss_log_stage1 = []
    t_start = pytime.time()

    for epoch in range(N_EPOCHS_STAGE1):

        eps = max(0.0, EPS_START * (1.0 - epoch / N_EPOCHS_STAGE1))

        # adversary step (maximise loss) every 2 generator steps
        if epoch % 2 == 0:
            loss_adv = loss_stage1(net, tf, eps, mu0_bias)
            test_opt.zero_grad()
            (-loss_adv).backward()
            test_opt.step()

        # generator step (minimise loss)
        loss_gen = loss_stage1(net, tf, eps, mu0_bias)
        gen_opt.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        loss_log_stage1.append(loss_gen.item())

        if epoch % 500 == 0 or epoch == N_EPOCHS_STAGE1 - 1:
            elapsed = pytime.time() - t_start
            eta     = elapsed / (epoch + 1) * (N_EPOCHS_STAGE1 - epoch - 1)
            print(f"    Epoch {epoch:5d}/{N_EPOCHS_STAGE1}  eps={eps:.4f}  "
                  f"loss={loss_gen.item():.3e}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    # ------------------------------------------------------------------
    # Stage 2: stationary fine-tuning
    # ------------------------------------------------------------------
    print(f"\n  --- Stage 2: stationary ({N_EPOCHS_STAGE2} epochs) ---")
    loss_log_stage2 = []
    t_start = pytime.time()

    for epoch in range(N_EPOCHS_STAGE2):

        # adversary step every 2 generator steps
        if epoch % 2 == 0:
            loss_adv = loss_stage2(net, tf, mu0_bias)
            test_opt.zero_grad()
            (-loss_adv).backward()
            test_opt.step()

        # generator step
        loss_gen = loss_stage2(net, tf, mu0_bias)
        gen_opt.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        loss_log_stage2.append(loss_gen.item())

        if epoch % 500 == 0 or epoch == N_EPOCHS_STAGE2 - 1:
            elapsed = pytime.time() - t_start
            eta     = elapsed / (epoch + 1) * (N_EPOCHS_STAGE2 - epoch - 1)
            print(f"    Epoch {epoch:5d}/{N_EPOCHS_STAGE2}  "
                  f"loss={loss_gen.item():.3e}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    # ------------------------------------------------------------------
    # Estimate final m* from a large batch at t=T
    # ------------------------------------------------------------------
    with torch.no_grad():
        n_eval = 10000
        x0_ev  = mu0_bias + SIGMA0 * torch.randn(n_eval, 1, device=device)
        r_ev   = torch.randn(n_eval, 1, device=device)
        t_ev   = T * torch.ones(n_eval, 1, device=device)
        x_ev   = net(t_ev, x0_ev, r_ev)
        m_star = x_ev.mean().item()

    print(f"\n  Final estimated m* = {m_star:.4f}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    fname = f'results_double_well/checkpoint_{side}.pt'
    torch.save({
        'state_dict'       : net.state_dict(),
        'loss_log_stage1'  : loss_log_stage1,
        'loss_log_stage2'  : loss_log_stage2,
        'm_star'           : m_star,
        'mu0_bias'         : mu0_bias,
        'hparams': {
            'KAPPA'            : KAPPA,
            'SIGMA2'           : SIGMA2,
            'T'                : T,
            'MU0'              : MU0,
            'SIGMA0'           : SIGMA0,
            'K'                : K,
            'M'                : M,
            'N_EPOCHS_STAGE1'  : N_EPOCHS_STAGE1,
            'N_EPOCHS_STAGE2'  : N_EPOCHS_STAGE2,
            'EPS_START'        : EPS_START,
        },
    }, fname)
    print(f"  Saved: {fname}")

    return net, loss_log_stage1, loss_log_stage2


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("1D Double-Well McKean-Vlasov — two-stage training")
    print(f"kappa={KAPPA}, sigma^2={SIGMA2}, T={T}")
    print(f"Stage 1: {N_EPOCHS_STAGE1} epochs, eps {EPS_START} -> 0")
    print(f"Stage 2: {N_EPOCHS_STAGE2} epochs, stationary")

    net_left,  loss1_left,  loss2_left  = train_one('left')
    net_right, loss1_right, loss2_right = train_one('right')

    print("\nBoth networks trained successfully.")
    print("Run plot_double_well.py to generate figures.")