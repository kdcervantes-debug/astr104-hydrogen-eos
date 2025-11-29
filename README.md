# ASTR 104 – Problem Set 4, Part 2: Hydrogen EOS

This repository contains Python code for Part 2 of ASTR 104 Problem Set 4.

It models a pure hydrogen gas with:
- Molecular hydrogen: H₂
- Neutral hydrogen: H I
- Ionized hydrogen: H II + e⁻

and computes:
- Equilibrium composition as a function of temperature T and pressure P
- Specific Helmholtz free energy and entropy for each species
- Total mixture entropy s_mix(T, P)
- Adiabatic temperature gradient ∇_ad(T, P)

## How to run
1. Create and activate a virtual environment (optional).
2. Install dependencies:

```bash
pip install -r requirements.txt
