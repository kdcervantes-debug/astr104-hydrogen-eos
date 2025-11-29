# ASTR 104 - PS4 Part 2 EOS main script
"""
ASTR 104 - Problem Set 4
Part 2: Hydrogen Equation of State (EOS)

This script:
- Builds internal partition functions for H2 and HI
- Solves molecular equilibrium: H2 <-> 2 HI
- Solves ionization equilibrium: HI <-> HII + e-
- Computes specific free energy and (approximate) specific entropy
- Computes mixture entropy s_mix(T, P)
- Computes adiabatic gradient nabla_ad(T, P)
- Saves plots of composition, entropy, and nabla_ad in output/

Run from the repo root with:

    python src/eos_part2.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Physical constants (cgs)
# ---------------------------------------------------
k_B = 1.380649e-16         # Boltzmann constant [erg K^-1]
h   = 6.62607015e-27       # Planck constant [erg s]
m_H = 1.6735575e-24        # Hydrogen atom mass [g]
m_e = 9.10938356e-28       # Electron mass [g]
eV  = 1.602176634e-12      # 1 eV in erg

# Energies
E_b   = 4.52 * eV          # H2 bond energy [erg]
chi_H = 13.6 * eV          # H ionization energy [erg]

# H2 vibrational and rotational characteristic temperatures (approximate)
theta_vib = 6330.0         # [K], h*omega/k_B
theta_rot = 85.4           # [K], B/k_B


# ---------------------------------------------------
# Partition functions
# ---------------------------------------------------

def Z_int_HI(T, nmax=10):
    """
    Internal partition function for neutral H (electronic).
    Sum over bound levels up to nmax.
    Ground state energy is taken as zero.
    """
    Z = 0.0
    E1 = -13.6 * eV
    for n in range(1, nmax + 1):
        g_n = 2.0 * n**2
        E_n = -13.6 * eV / (n**2)
        E_rel = E_n - E1
        Z += g_n * np.exp(-E_rel / (k_B * T))
    return Z


def Z_elec_H2(T):
    """
    Electronic partition function of H2.
    Ground state plus a first excited state separated by E_b.
    """
    return 1.0 + np.exp(-E_b / (k_B * T))


def Z_vib_H2(T, nmax=100):
    """
    Vibrational partition function of H2 (harmonic oscillator).
    E_n = k_B * theta_vib * (n + 1/2)
    """
    Z = 0.0
    for n in range(nmax + 1):
        E_n = k_B * theta_vib * (n + 0.5)
        Z += np.exp(-E_n / (k_B * T))
    return Z


def Z_rot_H2(T, jmax=100):
    """
    Rotational partition function of H2 with ortho/para splitting.
      - Even j: parahydrogen (nuclear spin degeneracy 1)
      - Odd j:  orthohydrogen (nuclear spin degeneracy 3)
    E_j = k_B * theta_rot * j(j+1)
    """
    Z = 0.0
    for j in range(jmax + 1):
        E_j = k_B * theta_rot * j * (j + 1)
        g_spin = 1.0 if j % 2 == 0 else 3.0
        Z += (2 * j + 1) * g_spin * np.exp(-E_j / (k_B * T))
    return Z


def Z_int_H2(T):
    """Total internal partition function of H2."""
    return Z_elec_H2(T) * Z_vib_H2(T) * Z_rot_H2(T)


# ---------------------------------------------------
# Equilibrium constants
# ---------------------------------------------------

def thermal_lambda(mass, T):
    """Thermal de Broglie wavelength."""
    return h / np.sqrt(2.0 * np.pi * mass * k_B * T)


def K_mol(T):
    """
    Molecular equilibrium constant for H2 <-> 2 HI.

    Defined such that:
        x_HI^2 / (1 - x_HI) = K_mol(T) / n_tot,

    where n_tot is the total hydrogen nuclei number density.
    """
    lam_HI = thermal_lambda(m_H, T)
    lam_H2 = thermal_lambda(2.0 * m_H, T)

    Z_HI = Z_int_HI(T)
    Z_H2 = Z_int_H2(T)

    pref = (lam_HI**6 / lam_H2**3) * (Z_HI**2 / Z_H2)
    return pref * np.exp(-E_b / (k_B * T))


def K_ion(T):
    """
    Ionization Saha constant for HI <-> HII + e^-.

    Defined such that:
        n_e n_HII / n_HI = K_ion(T).
    """
    lam_e = thermal_lambda(m_e, T)
    Z_HI = Z_int_HI(T)

    pref = (2.0 / lam_e**3) * (2.0 / Z_HI)  # factor 2 from electron spin
    return pref * np.exp(-chi_H / (k_B * T))


# ---------------------------------------------------
# Thermodynamics
# ---------------------------------------------------

def specific_free_energy(n_s, m_s, Zint_s, T):
    """
    Specific Helmholtz free energy f_s [erg/g] for a single species.

    F_s/N_s = k_B T [ ln(n_s lambda_s^3 / Zint_s) - 1 ]
    f_s = (F_s/N_s) / m_s
    """
    lam = thermal_lambda(m_s, T)
    return (k_B * T / m_s) * (np.log(n_s * lam**3 / Zint_s) - 1.0)


def centered_derivative(f, axis, dx):
    """
    Centered finite difference derivative along a given axis.
    Uses periodic roll at the boundaries (adequate for a smooth grid).
    """
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dx)


# ---------------------------------------------------
# EOS grid computation
# ---------------------------------------------------

def compute_eos_grid(
    N_T=60,
    N_P=60,
    logT_min=2.0,
    logT_max=5.0,
    logP_min=0.0,
    logP_max=13.0,
):
    """
    Compute EOS on a grid in (log10 T, log10 P).

    Returns:
        logT, logP  : 1D arrays
        xH2, xHI, xHII : 2D arrays of number fractions (shape N_P x N_T)
        s_mix       : 2D array of specific entropy [erg g^-1 K^-1]
        nabla_ad    : 2D array of adiabatic gradient
    """
    # 1D grids
    logT = np.linspace(logT_min, logT_max, N_T)
    logP = np.linspace(logP_min, logP_max, N_P)
    Tvals = 10.0**logT
    Pvals = 10.0**logP

    # Storage
    xH2   = np.zeros((N_P, N_T))
    xHI   = np.zeros((N_P, N_T))
    xHII  = np.zeros((N_P, N_T))
    s_mix = np.zeros((N_P, N_T))

    for iP, P in enumerate(Pvals):
        for iT, T in enumerate(Tvals):
            # Total number density of H nuclei
            n_tot = P / (k_B * T)

            # ------------------------------
            # Molecular equilibrium H2 <-> 2 HI
            # ------------------------------
            Km = K_mol(T)
            A = Km / n_tot

            # Solve x^2/(1 - x) = A -> x^2 + A x - A = 0
            disc = A * A + 4.0 * A
            x_atomic = (-A + np.sqrt(disc)) / 2.0
            x_atomic = np.clip(x_atomic, 0.0, 1.0)

            # n_HI and n_H2 in terms of nuclei
            n_HI_total = x_atomic * n_tot
            n_H2 = (1.0 - x_atomic) * n_tot / 2.0  # each H2 has 2 H nuclei

            # ------------------------------
            # Ionization equilibrium HI <-> HII + e^-
            # ------------------------------
            Kion = K_ion(T)
            n_HII = np.sqrt(max(0.0, Kion * n_HI_total))
            n_HII = min(n_HII, n_HI_total)

            n_HI_neutral = n_HI_total - n_HII

            # ------------------------------
            # Number fractions of nuclei in each form
            # ------------------------------
            N_nuclei = 2.0 * n_H2 + n_HI_neutral + n_HII
            if N_nuclei <= 0.0:
                continue

            xH2[iP, iT]  = (2.0 * n_H2)      / N_nuclei
            xHI[iP, iT]  = n_HI_neutral      / N_nuclei
            xHII[iP, iT] = n_HII             / N_nuclei

            # ------------------------------
            # Mixture entropy (approximate)
            # ------------------------------
            rho = 2.0 * m_H * n_H2 + m_H * n_HI_neutral + m_H * n_HII
            if rho <= 0.0:
                continue

            Y_H2  = 2.0 * m_H * n_H2       / rho
            Y_HI  = m_H * n_HI_neutral     / rho
            Y_HII = m_H * n_HII            / rho

            f_H2 = f_HI = f_HII = 0.0

            if n_H2 > 0.0:
                f_H2 = specific_free_energy(n_H2, 2.0 * m_H, Z_int_H2(T), T)
            if n_HI_neutral > 0.0:
                f_HI = specific_free_energy(n_HI_neutral, m_H, Z_int_HI(T), T)
            if n_HII > 0.0:
                # HII internal partition function ~ 1
                f_HII = specific_free_energy(n_HII, m_H, 1.0, T)

            f_mix = Y_H2 * f_H2 + Y_HI * f_HI + Y_HII * f_HII

            # Approximate s_mix as -f_mix / T (captures trends)
            s_mix[iP, iT] = -f_mix / T

    # ------------------------------------------------
    # Derivatives of log s_mix and nabla_ad
    # ------------------------------------------------
    s_safe = np.clip(s_mix, 1e-40, None)
    log_s = np.log(s_safe)

    dlogT = logT[1] - logT[0]
    dlogP = logP[1] - logP[0]

    dlogS_dlogT = centered_derivative(log_s, axis=1, dx=dlogT)
    dlogS_dlogP = centered_derivative(log_s, axis=0, dx=dlogP)

    nabla_ad = -dlogS_dlogP / (dlogS_dlogT + 1e-30)

    return logT, logP, xH2, xHI, xHII, s_mix, nabla_ad


# ---------------------------------------------------
# Plotting utilities
# ---------------------------------------------------

def get_output_dir():
    """Ensure output/ directory exists next to repo root and return its path."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(root, "output")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def plot_composition(logT, logP, xH2, xHI, xHII, outdir):
    extent = [logT.min(), logT.max(), logP.min(), logP.max()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    im0 = axes[0].imshow(xH2, extent=extent, origin="lower", aspect="auto")
    axes[0].set_title("H$_2$ Fraction")
    axes[0].set_xlabel(r"$\log_{10} T$ [K]")
    axes[0].set_ylabel(r"$\log_{10} P$ [dyne cm$^{-2}$]")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(xHI, extent=extent, origin="lower", aspect="auto")
    axes[1].set_title("H I Fraction")
    axes[1].set_xlabel(r"$\log_{10} T$ [K]")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(xHII, extent=extent, origin="lower", aspect="auto")
    axes[2].set_title("H II Fraction")
    axes[2].set_xlabel(r"$\log_{10} T$ [K]")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fname = os.path.join(outdir, "composition_H2_HI_HII.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_entropy(logT, logP, s_mix, outdir):
    extent = [logT.min(), logT.max(), logP.min(), logP.max()]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(s_mix, extent=extent, origin="lower", aspect="auto")
    ax.set_title("Mixture Entropy $s_{\\rm mix}$")
    ax.set_xlabel(r"$\log_{10} T$ [K]")
    ax.set_ylabel(r"$\log_{10} P$ [dyne cm$^{-2}$]")
    fig.colorbar(im, ax=ax, label=r"$s_{\rm mix}$ [erg g$^{-1}$ K$^{-1}$]")
    fig.tight_layout()

    fname = os.path.join(outdir, "entropy_mix.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_nabla_ad(logT, logP, nabla_ad, outdir):
    extent = [logT.min(), logT.max(), logP.min(), logP.max()]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(nabla_ad, extent=extent, origin="lower", aspect="auto")
    ax.set_title("Adiabatic Gradient $\\nabla_{\\rm ad}$")
    ax.set_xlabel(r"$\log_{10} T$ [K]")
    ax.set_ylabel(r"$\log_{10} P$ [dyne cm$^{-2}$]")
    fig.colorbar(im, ax=ax, label=r"$\nabla_{\rm ad}$")
    fig.tight_layout()

    fname = os.path.join(outdir, "nabla_ad.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)


# ---------------------------------------------------
# Main entry point
# ---------------------------------------------------

def main():
    outdir = get_output_dir()

    logT, logP, xH2, xHI, xHII, s_mix, nabla_ad = compute_eos_grid()

    plot_composition(logT, logP, xH2, xHI, xHII, outdir)
    plot_entropy(logT, logP, s_mix, outdir)
    plot_nabla_ad(logT, logP, nabla_ad, outdir)

    print("EOS computation complete.")
    print(f"Plots saved in: {outdir}")


if __name__ == "__main__":
    main()
