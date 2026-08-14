"""
Monoexponential phasor maths.

The phasor of a single-exponential decay of lifetime τ at excitation angular
frequency ω = 2πf lies on the universal semicircle::

    g = 1 / (1 + (ωτ)²)
    s = ωτ / (1 + (ωτ)²)

Everything that needs that relation — the lifetime tick marks drawn on the
phasor plot and the calibration factor derived from a reference dye — goes
through :func:`monoexp_phasor`, so the formula exists once.

The frequency must be the one the phasor data itself was reconstructed with,
i.e. ``repetition_rate`` from the file header (see ``core/io/loader.py``), not a
user-entered value; otherwise the reference points do not describe the data.
"""

import numpy as np


def monoexp_phasor(tau_s, f_rep_hz: float):
    """Phasor coordinates of a monoexponential decay.

    Args:
        tau_s: Lifetime in **seconds**; scalar or array.
        f_rep_hz: Excitation repetition rate in Hz.

    Returns:
        ``(g, s)`` — floats for a scalar *tau_s*, arrays otherwise. τ = 0 maps to
        (1, 0) and τ → ∞ to (0, 0), the two ends of the semicircle.
    """
    omega_tau = 2 * np.pi * f_rep_hz * np.asarray(tau_s, dtype=float)
    denom = 1.0 + omega_tau**2
    g = 1.0 / denom
    s = omega_tau / denom
    if np.isscalar(tau_s) or np.ndim(tau_s) == 0:
        return float(g), float(s)
    return g, s


def lifetime_ticks(
    f_rep_hz: float, tau_max_ns: int | None = None
) -> list[tuple[int, float, float]]:
    """Reference lifetimes to mark on the universal semicircle.

    Args:
        f_rep_hz: Excitation repetition rate in Hz.
        tau_max_ns: Largest lifetime to include, in ns. Defaults to half the
            laser period — beyond that the points crowd into the origin and
            stop being readable (13 ns at 40 MHz).

    Returns:
        ``[(tau_ns, g, s), …]`` for τ = 1 … *tau_max_ns* ns.
    """
    if f_rep_hz <= 0:
        return []
    if tau_max_ns is None:
        period_ns = 1e9 / f_rep_hz
        tau_max_ns = int(np.ceil(period_ns / 2))
    tau_max_ns = max(1, int(tau_max_ns))

    ticks = []
    for tau_ns in range(1, tau_max_ns + 1):
        g, s = monoexp_phasor(tau_ns * 1e-9, f_rep_hz)
        ticks.append((tau_ns, g, s))
    return ticks


def calibration_factor(
    tau_ns: float, measured: complex, f_rep_hz: float
) -> complex:
    """Factor that maps a *measured* reference phasor onto its true position.

    Applied as ``phasor * factor`` (see the Phasor panel), it corrects the
    instrument response using a dye of known lifetime *tau_ns*.

    Raises:
        ValueError: If *measured* is (numerically) zero, leaving the factor
            undefined.
    """
    if abs(measured) < 1e-12:
        raise ValueError(
            "Measured phasor is zero — cannot compute calibration factor."
        )
    g_th, s_th = monoexp_phasor(tau_ns * 1e-9, f_rep_hz)
    return complex(g_th, s_th) / measured
