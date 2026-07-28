# CC-PINN: Cross-Correlated Physics-Informed Neural Network for Robust Inverse Scattering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper **"Beyond Data-Physics Consistency: A Cross-Correlated Physics-Informed Neural Network for Robust Inverse Scattering"**.

CC-PINN introduces a novel physically-informed learning framework to solve highly nonlinear and ill-posed Electromagnetic Inverse Scattering Problems (ISPs). By bridging the gap between data-physics consistency and contrast source inversion, CC-PINN achieves state-of-the-art reconstruction accuracy and enhanced robustness on high-contrast and lossy dielectric targets.

## Key Innovations

1. **Cross-Correlated Physics Loss:** Breaks the traditional decoupling between contrast source and permittivity optimization, guiding the network away from severe local minima.
2. **Zero-Padding 2D-FFT Acceleration:** Converts dense matrix-vector multiplications of the Volume Integral Equation (VIE) into spatial linear convolutions, reducing the computational complexity of the Green's function integration from $\mathcal{O}(N^4)$ to $\mathcal{O}(N^2\log N)$.
3. **Weight-Normalized Fourier Feature MLP:** Effectively overcomes the "spectral bias" of standard MLPs, enabling the network to capture high-frequency physical boundaries and local high-contrast jumps with extreme stability.

## Implemented Algorithms

To ensure a fair and comprehensive comparison, this repository includes the implementation of four algorithms within a unified testing framework:
* **CC-PINN:** The proposed Cross-Correlated PINN.
* **Data+State-PINN:** Conventional PINN relying solely on independent data and state equation residuals.
* **ES-PINN (Exact-Solver PINN):** A strictly physics-constrained PINN that performs explicit matrix inversion at each step.
* **CC-CSI:** The classical Cross-Correlated Contrast Source Inversion (traditional iterative optimization method).

## Datasets Evaluated
The code natively supports automated robustness testing on various datasets:
* **Synthetic Data:** "Austria" profile and "Bowtie-Cross" targets (varying relative permittivity $\varepsilon_r \in [4, 7]$, lossy media, and different noise levels: 20dB, 10dB, 0dB SNR).
* **Measured Data:** The public `FoamTwinDielTM` dataset provided by the Institut Fresnel (supports both Frequency-Hopping and Simultaneous Multi-Frequency processing strategies).

## Dependencies

* Python 3.8+
* PyTorch (CUDA supported and highly recommended)
* NumPy
* SciPy
* Matplotlib

Install the required packages using:
```bash
pip install torch numpy scipy matplotlib
