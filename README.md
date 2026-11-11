# Deep Hedging for a Two-Asset Product Option

**Python · TensorFlow / Keras · Derivatives · Monte Carlo Simulation**

## Overview

This project develops a neural-network hedging strategy for a European call on the normalized product of two correlated assets. Both asset prices follow geometric Brownian motion, and the hedge is rebalanced at 100 discrete time steps over a one-year horizon.

The objective is to learn the **initial option premium** and the **positions in both underlying assets** that minimize the mean squared terminal replication error.

## Methodology

- **Path simulation:** Generate 100,000 correlated two-asset trajectories for training and 10,000 independent trajectories for evaluation.
- **Deep hedging:** A shared feed-forward network takes time and both asset prices as inputs and outputs two hedge ratios; a separate network learns the initial premium.
- **Self-financing portfolio:** Update portfolio wealth using hedge gains and interest on the cash position at each rebalancing date.
- **Model evaluation:** Examine train/test hedging-error distributions, learned delta profiles, and two-dimensional hedge surfaces.
- **Pricing benchmark:** Compare the learned premium with a risk-neutral Monte Carlo price.

## Results

For the parameter configuration in the notebook ($\sigma_1=0.22$, $\sigma_2=0.30$, $\rho=0.50$, $r=0.04$), the saved experiment reports:

| Metric | Result |
|---|---:|
| Learned premium | 0.2439 |
| Monte Carlo reference price | 0.2454 |
| Test hedging-error mean | −0.00126 |
| Test hedging-error standard deviation | 0.02258 |
| Test hedging-error MSE | 0.000511 |

The test error distribution remains close to the training distribution for simulated paths generated under the same dynamics. The model is evaluated **without transaction costs, liquidity constraints, or changes in the assumed stochastic dynamics**. Although Monte Carlo is used as the numerical reference in the notebook, this particular product option also admits analytical valuation under the joint lognormal assumptions.

## Repository contents

- `deep_hedging_2d.ipynb` — Simulation, model architecture, training, pricing, hedge analysis, and visualizations.
- `requirements.txt` — Python dependencies.

## Running the notebook

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Open `deep_hedging_2d.ipynb` in Jupyter or Google Colab and run the cells in order. The project generates its own synthetic data and does not require external datasets.

The original configuration trains on 100,000 paths for 20 epochs and simulates 500,000 additional paths for Monte Carlo valuation. A GPU is helpful; for a faster local demonstration, reduce `N_train`, `N_test`, `N_mc`, or the number of epochs. Results may change with sample size and random initialization.
