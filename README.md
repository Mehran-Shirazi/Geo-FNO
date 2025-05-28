# Geometry-Aware Fourier Neural Operator (Geo-FNO) – PyTorch Lightning Version

This repository contains a refactored version of the original [Geo-FNO](https://github.com/zongyi-li/Geo-FNO) codebase, implemented with **[PyTorch Lightning](https://www.pytorchlightning.ai/)** to improve modularity, scalability, and training abstraction.

> ✅ This work builds upon the original implementation by **Zongyi Li**, published in the paper:  
> 📄 *"Fourier Neural Operator with Geometry-aware Input Mapping"*  
> [arXiv:2207.05209](https://arxiv.org/abs/2207.05209)

---

## 🔧 What is Geo-FNO?

Geo-FNO is a deep learning-based surrogate model for solving partial differential equations (PDEs) on **arbitrary geometries**. It extends the classic Fourier Neural Operator (FNO) by learning a geometric mapping from irregular domains to a latent uniform grid, where FFT-based convolution is performed efficiently.

Geo-FNO supports:
- Arbitrary input domains (point clouds, meshes, design parameters)
- Multiple physics (Elasticity, Plasticity, Euler, Navier-Stokes)
- Forward modeling and inverse design
- Up to **10–50× speedup** over traditional solvers

---

## ✨ What's New in This Version?

This version refactors the original PyTorch code using **PyTorch Lightning**, enabling:

- Cleaner training/validation loops
- Easier experiment management with callbacks and loggers
- Better reproducibility

---

## 📦 Requirements

- `pytorch >= 1.8.0`
- `pytorch-lightning >= 2.0.0`
- `h5py`, `numpy`, `matplotlib`, etc.


<!---
Install dependencies via:

```bash
pip install -r requirements.txt
-->
