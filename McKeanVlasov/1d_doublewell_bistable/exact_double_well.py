"""
exact_double_well.py
--------------------
Utility functions for the 1D double-well McKean-Vlasov problem.

Effective potential:
    Phi(x; m*) = x^4/4 + (kappa-1)/2 * x^2 - kappa * m* * x

Stationary Gibbs density:
    rho*(x; m*) propto exp(-beta * Phi(x; m*)),  beta = 2/sigma^2

Self-consistency condition:
    m* = E_{rho*(.; m*)}[x]
"""

import numpy as np

# -----------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------
_X_MIN, _X_MAX, _N = -4.0, 4.0, 2000


def _grid():
    xs = np.linspace(_X_MIN, _X_MAX, _N)
    dx = xs[1] - xs[0]
    return xs, dx


# -----------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------

def gibbs_density(m, kappa, sigma2, xs=None):
    """
    Normalised Gibbs density on a grid.

    Parameters
    ----------
    m      : float   - self-consistent mean m*
    kappa  : float   - interaction strength (need kappa < 1 for bistability)
    sigma2 : float   - diffusion coefficient sigma^2
    xs     : array   - evaluation grid (uses internal default if None)

    Returns
    -------
    xs   : (N,) array
    rho  : (N,) array  - normalised density
    """
    if xs is None:
        xs, dx = _grid()
    else:
        dx = xs[1] - xs[0]
    beta = 2.0 / sigma2
    log_rho = -beta * (xs**4 / 4.0 + (kappa - 1.0) / 2.0 * xs**2 - kappa * m * xs)
    log_rho -= log_rho.max()
    rho = np.exp(log_rho)
    rho /= rho.sum() * dx
    return xs, rho


def effective_potential(m, kappa, sigma2, xs=None):
    """
    Effective potential Phi(x; m*) = x^4/4 + (kappa-1)/2 * x^2 - kappa*m*x.

    Returns
    -------
    xs  : (N,) array
    phi : (N,) array
    """
    if xs is None:
        xs, _ = _grid()
    phi = xs**4 / 4.0 + (kappa - 1.0) / 2.0 * xs**2 - kappa * m * xs
    return xs, phi


def _self_consistency_residual(m, kappa, sigma2):
    """E_{rho*(m)}[x] - m"""
    xs, dx = _grid()
    _, rho = gibbs_density(m, kappa, sigma2, xs)
    return float((xs * rho).sum() * dx) - m


def self_consistent_means(kappa, sigma2, n_search=4000):
    """
    Find all fixed points of  m* = E_{rho*(m*)}[x]  by bisection.

    Returns
    -------
    roots : list of (m*, stability) tuples
            stability is 'stable' or 'unstable'
    """
    ms = np.linspace(-2.5, 2.5, n_search)
    f = np.array([_self_consistency_residual(m, kappa, sigma2) for m in ms])

    roots = []
    for i in range(len(ms) - 1):
        if f[i] * f[i + 1] < 0:
            a, b = ms[i], ms[i + 1]
            for _ in range(60):
                mid = (a + b) / 2.0
                if (_self_consistency_residual(mid, kappa, sigma2) *
                        _self_consistency_residual(a, kappa, sigma2)) < 0:
                    b = mid
                else:
                    a = mid
            roots.append((a + b) / 2.0)

    # Classify stability: residual slope < 0 at root => stable
    classified = []
    eps = 0.005
    for r in roots:
        df = (_self_consistency_residual(r + eps, kappa, sigma2) -
              _self_consistency_residual(r - eps, kappa, sigma2)) / (2 * eps)
        stability = 'stable' if df < 0 else 'unstable'
        classified.append((r, stability))

    return classified
