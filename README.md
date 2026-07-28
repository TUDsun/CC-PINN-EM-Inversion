# CC-PINN: Cross-Correlated Physics-Informed Neural Network for Robust Inverse Scattering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.01851-b31b1b.svg)](https://arxiv.org/abs/2605.01851)

This repository contains the official PyTorch implementation of the paper **"Beyond Data-Physics Consistency: A Cross-Correlated Physics-Informed Neural Network for Robust Inverse Scattering"**.

CC-PINN introduces a novel physically-informed learning framework to solve highly nonlinear and ill-posed Electromagnetic Inverse Scattering Problems (ISPs). By bridging the gap between data-physics consistency and contrast source inversion, CC-PINN achieves state-of-the-art reconstruction accuracy and remarkable robustness on high-contrast and lossy dielectric targets.

## Key Innovations

1. **Cross-Correlated Physics Loss:** Breaks the traditional decoupling between contrast source and permittivity optimization, guiding the network away from severe local minima.
2. **Zero-Padding 2D-FFT Acceleration:** Converts dense matrix-vector multiplications of the Volume Integral Equation (VIE) into spatial linear convolutions, reducing the computational complexity of the Green's function integration from $\mathcal{O}(N^4)$ to $\mathcal{O}(N^2\log N)$.
3. **Weight-Normalized Fourier Feature MLP:** Effectively overcomes the "spectral bias" of standard MLPs, enabling the network to capture high-frequency physical boundaries and local high-contrast jumps with extreme stability.

## Implemented Algorithms

To ensure a fair and comprehensive comparison, this repository includes the implementation of four algorithms within a unified testing framework:
* **CC-PINN:** The proposed Cross-Correlated PINN Electromagnetic inversion algorithm.
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
```

## How to Run

The main script is highly automated. It executes 11 independent runs (with different random seeds) for all 4 algorithms on the specified datasets, automatically computing metrics (PSNR, SSIM, Time) and plotting convergence curves and boxplots.

```bash
python robustness_test_CC-PINN_ES-PINN_CC-CSI-Time-PSNR_autoRun_optimized_Epochs.py
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

## ✉️ Contact
For any questions regarding the code or the paper, please open an issue in this repository or contact:
**Shilong Sun** - [sunshilong@nudt.edu.cn](mailto:sunshilong@nudt.edu.cn)
