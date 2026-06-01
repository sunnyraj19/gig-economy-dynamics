# Gig Economy Financial Dynamics

## Overview
This project applies nonlinear dynamic systems theory to analyze the financial stability of gig economy workers in India. By modeling workers as dynamic systems, we identify "tipping points" where income volatility leads to debt traps.

## Methodology
1. **Data Fusion**: Merging static worker profiles with transaction histories.
2. **State Reconstruction**: Defining state variables $W$ (Wealth Stability) and $D$ (Debt Pressure).
3. **Phase Space Partitioning**: Using K-Means clustering to identify distinct financial archetypes.
4. **Stability Analysis**: Jacobian estimation and eigenvalue analysis to determine local stability of each archetype.

## Directory Structure
- `src/`: Modular Python code for data processing and modeling.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `data/`: Raw and processed datasets.
- `results/`: Generated figures and model outputs.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `gig_workers.csv` and `transactions.csv` in `data/raw/`.
3. Run `notebooks/01_eda.ipynb`.

## Key Findings
- Identified 4 distinct financial archetypes among 120,000 workers.
- **Archetype 3** was found to be financially unstable, with a negative debt sensitivity of -0.0019, indicating a high risk of falling into a debt trap.


