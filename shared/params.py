"""
================================================================================
MODEL PARAMETERS
================================================================================

Two parameter sets for the pangenome bet-hedging model:

THEORY_PARAMS : used by Figures 1-5 and supplementary figures S1-S2.
    s = 0.3, c = 0.02, h = 0.001, delta = 0.001, p = 0.01
    These illustrate the qualitative mechanisms clearly in simulations.

EMPIRICAL_PARAMS : used by empirical reanalyses (Figures 4-5, SI S4-S11).
    s = 0.01, c = 0.0002
    Biologically realistic estimates for the E. coli pangenome.

================================================================================
"""

# ── Theory figures (Figs 1-5) ────────────────────────────────────────────────

THEORY_PARAMS = {
    's': 0.3,       # Selective benefit when gene is needed (30%)
    'c': 0.02,      # Carriage cost when gene is not needed (2%)
    'h': 0.001,     # HGT rate per generation (0.1%)
    'delta': 0.001, # Gene loss rate per generation (0.1%)
    'p': 0.01,      # Probability environment favours the gene (1%)
}


# ── Empirical analyses (Figs 4-5) ───────────────────────────────────────────

EMPIRICAL_PARAMS = {
    's': 0.01,      # Selective benefit (~1%)
    'c': 0.0002,    # Carriage cost (~0.02%)
}


# ── Backward-compatible aliases ──────────────────────────────────────────────
# Scripts that did `from params import PARAMS` still work.

PARAMS = THEORY_PARAMS


# ── Derived quantities ───────────────────────────────────────────────────────

def complexity_threshold(s=None, c=None):
    """Critical environmental complexity E_crit = s/c.

    Above this, single-genome strategies cannot maintain full coverage.
    """
    s = s if s is not None else EMPIRICAL_PARAMS['s']
    c = c if c is not None else EMPIRICAL_PARAMS['c']
    return s / c


def expected_equilibrium_freq(h=None, c=None, delta=None):
    """Equilibrium frequency under HGT-selection-loss balance.

    f_eq = h / (c + delta + h)  for a net-deleterious gene.
    """
    h = h if h is not None else THEORY_PARAMS['h']
    c = c if c is not None else THEORY_PARAMS['c']
    delta = delta if delta is not None else THEORY_PARAMS['delta']
    return h / (c + delta + h)
