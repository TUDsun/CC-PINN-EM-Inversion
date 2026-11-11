# CC-PINN: Cross-Correlated Physics-Informed Neural Network for Robust Inverse Scattering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01851-b31b1b.svg)](https://arxiv.org/abs/2605.01851)

This repository contains the official PyTorch implementation of the paper **"Beyond Data-Physics Consistency: A Cross-Correlated Physics-Informed Neural Network for Robust Inverse Scattering"**.

CC-PINN introduces a novel physically-informed learning framework to solve highly nonlinear and ill-posed Electromagnetic Inverse Scattering Problems (ISPs). By bridging the gap between data-physics consistency and contrast source inversion, CC-PINN achieves state-of-the-art reconstruction accuracy and remarkable robustness on high-contrast and lossy dielectric targets.

## Key Innovations

1. **Cross-Correlated Physics Loss:** Breaks the traditional decoupling between contrast source and permittivity optimization, guiding the network away from severe local minima.
2. **Weight-Normalized Fourier Feature MLP:** Effectively overcomes the "spectral bias" of standard MLPs, enabling the network to capture high-frequency physical boundaries and local high-contrast jumps with extreme stability.

## Features
1. **Zero-Padding 2D-FFT Acceleration:** Converts dense matrix-vector multiplications of the Volume Integral Equation (VIE) into spatial linear convolutions, reducing the computational complexity of the Green's function integration from $\mathcal{O}(N_\text{g}^2)$ to $\mathcal{O}(N_\text{g}\log N_\text{g})$ (Assuming the domain of interest is discretized into an $N \times N$ grid, the total number of unknown pixels is $N_\text{g} = N^2$).

## Implemented Algorithms

To ensure a fair and comprehensive comparison, this repository includes the implementation of four algorithms within a unified testing framework:
* **CC-PINN:** The proposed Cross-Correlated PINN Electromagnetic inversion algorithm.
* **Data+State-PINN:** Conventional PINN relying solely on independent data and state equation residuals.
* **ES-PINN (Exact-Solver PINN):** A strictly physics-constrained PINN that performs explicit matrix inversion at each step.
* **CC-CSI:** The classical Cross-Correlated Contrast Source Inversion (traditional iterative optimization method).

**Supplementary Script for Baseline Selection:**
* `CC-CSI_CSI_DBIM-Time-PSNR_autoRun.py`: A standalone evaluation script comparing the reconstruction accuracy (PSNR) and computational time among **CC-CSI, conventional CSI, and DBIM**. The empirical results explicitly demonstrate the superiority of CC-CSI, rigorously justifying its selection as the representative traditional deterministic inversion baseline in our manuscript.

## Important Note on Learning Rate Tuning & Reproduction

**Note for Reproduction and Comparison with the Manuscript**

In the published manuscript, the synthetic simulation experiments were conducted using the initial learning rate configuration from the earlier submission round. In this updated codebase, a more balanced and refined learning rate configuration has been adopted:
```python
LR_NET_BASE, LR_W_BASE = (1e-3, 1e-3) if "Fresnel" in FILE_NAME else (1e-4, 1e-4)
ETA_NET_MIN = LR_NET_BASE
ETA_W_MIN = LR_W_BASE
```

When running the code with these refined parameters, users may observe the following two phenomena:
1. **Substantial Robustness Gain over CC-CSI:** Both `Data+State-PINN` and `CC-PINN` exhibit convergence robustness that significantly outperforms the classical deterministic solver `CC-CSI`.
2. **Contrast Limit on Idealized Synthetic Data:** Under this specific learning rate configuration, `Data+State-PINN` can achieve a contrast retrieval limit slightly higher than that of `CC-PINN` on idealized synthetic benchmarks.

**Scientific Clarification on Inversion Robustness:**
We emphasize that Phenomenon 2 **does not** imply that `Data+State-PINN` is intrinsically more robust than `CC-PINN` in realistic inverse problems. 
* On experimental measured data (the Institut Fresnel dataset), **CC-PINN demonstrates vastly superior robustness and reconstruction fidelity compared to Data+State-PINN**.
* **Physical Mechanism:** Synthetic simulations operate under idealized 2D assumptions with pure additive noise. In contrast, real-world measured data inevitably involve severe unmodeled physical factors, including 3D wave propagation effects, incident field modeling mismatches, antenna radiation pattern distortions, and mutual coupling. Without the closed-loop cross-correlation constraint, `Data+State-PINN` easily overfits to these unmodeled systematic discrepancies and gets trapped in non-physical local minima from the earliest optimization stages. `CC-PINN`, by enforcing cross-validation between the inferred contrast, internal fields, and observation data, regularizes the optimization trajectory and prevents catastrophic local-minima traps. Therefore, **CC-PINN possesses substantially higher robustness in practical engineering applications**.

**Why are the simulation results in the manuscript not updated?**
Since the manuscript received a **Minor Revision** decision, the reported synthetic simulation results strictly preserve the original benchmark parameters established during early review stages to maintain revision consistency. Only the experimental Fresnel dataset was re-evaluated and updated using the refined learning rate configuration. This note is provided to ensure full transparency and prevent any confusion when reproducing the results against the manuscript.

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
```

## How to Run

The main script is highly automated. It executes 11 independent runs (with different random seeds) for all 4 algorithms on the specified datasets, automatically computing metrics (PSNR, SSIM, Time) and plotting convergence curves and boxplots.

```bash
python CC-PINN.py
```

### Outputs
After the execution, the results will be saved in the configured output directory (e.g., `robustness_test_sim/` or `robustness_test_hop/`). The generated files include:
* `PSNR_Convergence_eps/sig.pdf`: Step-vs-PSNR convergence curves with standard deviation bands.
* `Time_PSNR_Convergence_eps/sig.pdf`: Time-vs-PSNR convergence curves.
* `PSNR_Boxplots_eps/sig.pdf`: Statistical boxplots of the final PSNR over 11 runs.
* `reconstruction_final_eps/sig.pdf`: 2D reconstructed images of relative permittivity and conductivity.
* `metrics.npz` and `.pth` weight files for each run.

## Citation

**This paper is currently under review at IEEE Transactions on Antennas and Propagation (IEEE TAP).** You can access the preprint on arXiv: [https://arxiv.org/abs/2605.01851](https://arxiv.org/abs/2605.01851).

If you find this code or our proposed CC-PINN framework useful in your research, please consider citing our work:

```bibtex
@misc{sun2026beyond,
  title={Beyond Data-Physics Consistency: A Cross-Correlated Physics-Informed Neural Network for Robust Inverse Scattering}, 
  author={Shilong Sun},
  year={2026},
  eprint={2605.01851},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.01851}
}
```
*(Note: Please update the citation information once the paper is officially accepted and published.)*

## Contact
For any questions regarding the code or the paper, please open an issue in this repository or contact:
**Shilong Sun** - [sunshilong@nudt.edu.cn](mailto:sunshilong@nudt.edu.cn)
