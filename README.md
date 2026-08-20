# Cultural competition between scientific and alternative medicine

Simulation and figure-generation code for a mean-field model of cultural
competition between **scientific medicine** (`M`) and an **alternative
treatment** (`A`), coupled to a **non-infectious** disease.

Each script is **self-contained** (model parameters at the top, no shared
imports) and writes one figure as a vector PDF.

## Model

State vector `(M, I_M, I_A)`, where `M` is the fraction of the population
adopting scientific medicine (`1 - M` adopts the alternative), and `I_M`, `I_A`
are the ill fractions within each group. The disease is non-infectious:
healthy individuals fall ill at rate `beta` and recover at `gamma_M` (medicine)
or `gamma_A` (alternative), with `gamma_M > gamma_A`. Cultural change follows a
replicator equation whose attractiveness is based on observed health trends,
amplified by group-specific reporting rates `r_M`, `r_A`.

The main model produces a continuous, neutrally stable manifold of coexisting
equilibria (path dependence); an alternative "absolute-recovery" formulation
instead produces winner-takes-all bistability.

> Note: in the code the alternative group is denoted with the subscript `H`
> (`gamma_H`, `r_H`) for historical reasons; it corresponds to `A` in the paper.

## Contents

**Main-text figures**

| Script | Figure | Content |
|---|---|---|
| `fig1_transient.py` | Fig. 1 | Transient dynamics: belief fractions, disease burdens, and the `M`–`I_M` projection (monotone, non-oscillatory). |
| `fig2_variety_hysteresis.py` | Fig. 2 | The continuous variety of equilibria (two projections with trajectories freezing on the manifold) and path-dependent hysteresis under transient campaigns. |
| `fig3_sensitivity.py` | Fig. 3 | Sensitivity analysis: total endemic burden vs. incidence `beta`, and the frozen state `M(inf)` vs. the advocacy ratio `r_A/r_M` for two initial regimes. |

**Supplementary figures**

| Script | Content |
|---|---|
| `Fig_SuppInf_AlternativeModel_1.py` | Basin of attraction of the alternative (absolute-recovery) model (single panel). |
| `Fig_SuppInf_AlternativeModel_2.py` | Phase portrait of the alternative model: nullclines, escape separatrix, and the interior saddle (two panels). |
| `Fig_SuppInf_sensitivity.py` | Sensitivity of the alternative model (parameter heatmap and advocacy sweep). |

## Requirements

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

## Usage

Run any script from its folder; each writes its PDF to the current directory:

```bash
python fig1_transient.py
python fig2_variety_hysteresis.py
python fig3_sensitivity.py
```
