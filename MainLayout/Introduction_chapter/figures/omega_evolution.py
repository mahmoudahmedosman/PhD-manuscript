"""
Evolution of Omega_m, Omega_r, Omega_Lambda as functions of
scale factor a and cosmic time t.

Uses Planck 2018 best-fit parameters:
  Omega_m0   = 0.3111
  Omega_r0   = 9.24e-5  (photons + neutrinos)
  Omega_Lambda0 = 0.6889
  H0         = 67.66 km/s/Mpc
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint

# ── Cosmological parameters (Planck 2018) ───────────────────────────────────
H0_si    = 67.66e3 / 3.0857e22   # s^-1
Omega_m0 = 0.3111
Omega_r0 = 9.24e-5
Omega_L0 = 1.0 - Omega_m0 - Omega_r0   # flat universe

# ── Derived quantities ───────────────────────────────────────────────────────
def E(a):
    """Dimensionless Hubble parameter E(a) = H(a)/H0."""
    return np.sqrt(Omega_r0 * a**-4 + Omega_m0 * a**-3 + Omega_L0)

def Omega_m(a):
    return Omega_m0 * a**-3 / E(a)**2

def Omega_r(a):
    return Omega_r0 * a**-4 / E(a)**2

def Omega_L(a):
    return Omega_L0 / E(a)**2

# ── Cosmic time t(a) via numerical integration ───────────────────────────────
Gyr_per_s = 1.0 / (3.1558e16)   # 1 Gyr in seconds → conversion factor

def t_Gyr(a_val):
    """Cosmic time in Gyr at scale factor a_val."""
    integrand = lambda ap: 1.0 / (ap * H0_si * E(ap))
    val, _ = quad(integrand, 1e-10, a_val, limit=200)
    return val * Gyr_per_s

# ── Build arrays ─────────────────────────────────────────────────────────────
a_arr = np.logspace(-4, 0.7, 1500)   # a from 1e-4 to ~5

om_arr = Omega_m(a_arr)
or_arr = Omega_r(a_arr)
ol_arr = Omega_L(a_arr)

# Cosmic time for each a (vectorised via list comp — fast enough)
t_arr = np.array([t_Gyr(a) for a in a_arr])

# Today's values
a_today = 1.0
t_today = t_Gyr(a_today)

# ── Matplotlib style ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "font.family":        "serif",
    "font.size":          13,
    "axes.labelsize":     14,
    "axes.titlesize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "0.7",
    "axes.linewidth":     1.2,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# Colour palette (colour-blind friendly)
C_m = "#E07B54"    # warm orange  → matter
C_r = "#5B8DB8"    # steel blue   → radiation
C_L = "#6AAB6E"    # muted green  → Lambda
C_today = "#333333"

# ── Figure 1: vs scale factor ─────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

ax1.plot(a_arr, om_arr, color=C_m, lw=2.0, label=r"$\Omega_{\rm m}(a)$")
ax1.plot(a_arr, or_arr, color=C_r, lw=2.0, label=r"$\Omega_{\rm r}(a)$")
ax1.plot(a_arr, ol_arr, color=C_L, lw=2.0, label=r"$\Omega_{\Lambda}(a)$")

# Equality lines
a_eq_rm = Omega_r0 / Omega_m0          # radiation–matter equality
a_eq_mL = (Omega_m0 / Omega_L0)**(1/3) # matter–Lambda equality

ax1.axvline(a_eq_rm, color=C_r,   lw=1.0, ls="--", alpha=0.6)
ax1.axvline(a_eq_mL, color=C_L,   lw=1.0, ls="--", alpha=0.6)
ax1.axvline(a_today, color=C_today, lw=1.2, ls=":",  alpha=0.8)

ax1.text(a_eq_rm * 1.15, 0.72,
         r"$a_{\rm eq}^{r{\rm -}m}$", color=C_r, fontsize=10, va="center")
ax1.text(a_eq_mL * 0.50, 0.95,
         r"$a_{\rm eq}^{m{\rm -}\Lambda}$", color=C_L, fontsize=10, va="center")
ax1.text(a_today * 1.08, 0.40,
         r"$a_0 = 1$", color=C_today, fontsize=10, va="center")

ax1.set_xscale("log")
ax1.set_xlim(a_arr[0], a_arr[-1])
ax1.set_ylim(-0.02, 1.05)
ax1.set_xlabel(r"Scale factor $a$")
ax1.set_ylabel(r"Density parameter $\Omega_i(a)$")
ax1.set_title(r"Evolution of density parameters with scale factor")
ax1.legend(loc="center left", ncol=1)
ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

fig1.tight_layout()
fig1.savefig("omega_vs_a.pdf")
print("Saved omega_vs_a.pdf")

# ── Figure 2: vs cosmic time ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

ax2.plot(t_arr, om_arr, color=C_m, lw=2.0, label=r"$\Omega_{\rm m}(t)$")
ax2.plot(t_arr, or_arr, color=C_r, lw=2.0, label=r"$\Omega_{\rm r}(t)$")
ax2.plot(t_arr, ol_arr, color=C_L, lw=2.0, label=r"$\Omega_{\Lambda}(t)$")

t_eq_rm = t_Gyr(a_eq_rm)
t_eq_mL = t_Gyr(a_eq_mL)

ax2.axvline(t_eq_rm,  color=C_r,    lw=1.0, ls="--", alpha=0.6)
ax2.axvline(t_eq_mL,  color=C_L,    lw=1.0, ls="--", alpha=0.6)
ax2.axvline(t_today,  color=C_today, lw=1.2, ls=":",  alpha=0.8)

ax2.text(t_eq_rm * 2.5, 0.72,
         r"$t_{\rm eq}^{r{\rm -}m}$", color=C_r, fontsize=10, va="center")
ax2.text(t_eq_mL * 1.15, 0.72,
         r"$t_{\rm eq}^{m{\rm -}\Lambda}$", color=C_L, fontsize=10, va="center")
ax2.text(t_today * 1.04, 0.40,
         r"$t_0$", color=C_today, fontsize=10, va="center")

ax2.set_xscale("log")
ax2.set_xlim(t_arr[0], t_arr[-1])
ax2.set_ylim(-0.02, 1.05)
ax2.set_xlabel(r"Cosmic time $t$ [Gyr]")
ax2.set_ylabel(r"Density parameter $\Omega_i(t)$")
ax2.set_title(r"Evolution of density parameters with cosmic time")
ax2.legend(loc="center left", ncol=1)
ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))

fig2.tight_layout()
fig2.savefig("omega_vs_t.pdf")
print("Saved omega_vs_t.pdf")

# ── Figure 3: combined two-panel figure (manuscript-ready) ───────────────────
fig3, (axA, axB) = plt.subplots(2, 1, figsize=(12, 4.5),
                                  sharey=True, constrained_layout=True)

for ax, xdata, xlabel, xeq_rm, xeq_mL, xtod, xlim, xtod_label in [
    (axA, a_arr, r"Scale factor $a$",
     a_eq_rm, a_eq_mL, a_today,
     (a_arr[0], a_arr[-1]), r"$a_0 = 1$"),
    (axB, t_arr, r"Cosmic time $t$ [Gyr]",
     t_eq_rm, t_eq_mL, t_today,
     (t_arr[0], t_arr[-1]), r"$t_0$"),
]:
    ax.plot(xdata, om_arr, color=C_m, lw=2.0, label=r"$\Omega_{\rm m}$")
    ax.plot(xdata, or_arr, color=C_r, lw=2.0, label=r"$\Omega_{\rm r}$")
    ax.plot(xdata, ol_arr, color=C_L, lw=2.0, label=r"$\Omega_{\Lambda}$")

    ax.axvline(xeq_rm, color=C_r,    lw=1.0, ls="--", alpha=0.6)
    ax.axvline(xeq_mL, color=C_L,    lw=1.0, ls="--", alpha=0.6)
    ax.axvline(xtod,   color=C_today, lw=1.2, ls=":",  alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(xlabel)
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.legend(loc="center left")

axA.set_ylabel(r"Density parameter $\Omega_i$")

# Shared annotation — radiation–matter equality
for ax, xeq, label, c in [
    (axA, a_eq_rm, r"$a_{\rm eq}^{r{\rm -}m}$", C_r),
    (axA, a_eq_mL, r"$a_{\rm eq}^{m{\rm -}\Lambda}$", C_L),
    (axA, a_today, r"$a_0$", C_today),
    (axB, t_eq_rm, r"$t_{\rm eq}^{r{\rm -}m}$", C_r),
    (axB, t_eq_mL, r"$t_{\rm eq}^{m{\rm -}\Lambda}$", C_L),
    (axB, t_today, r"$t_0$", C_today),
]:
    ax.text(xeq * 1.3, 0.78, label, color=c, fontsize=9.5, va="center")

fig3.savefig("Omega_combined.pdf")
print("Saved omega_combined.pdf")
print(f"\nCosmological parameters used:")
print(f"  Omega_m0     = {Omega_m0}")
print(f"  Omega_r0     = {Omega_r0}")
print(f"  Omega_Lambda = {Omega_L0:.4f}")
print(f"  H0           = 67.66 km/s/Mpc")
print(f"\nDerived quantities:")
print(f"  a_eq (r-m)   = {a_eq_rm:.2e}")
print(f"  a_eq (m-L)   = {a_eq_mL:.4f}")
print(f"  t0 (today)   = {t_today:.2f} Gyr")
print(f"  t_eq (r-m)   = {t_eq_rm*1e3:.2f} Myr")
print(f"  t_eq (m-L)   = {t_eq_mL:.2f} Gyr")
