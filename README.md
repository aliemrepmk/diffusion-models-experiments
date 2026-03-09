# Diffusion Models Experiments

This repository contains experiments and educational implementations of **diffusion-based generative models** using **PyTorch**.

The goal is to understand how diffusion models work by implementing them from scratch and visualizing their behavior.

---

## Implementations

### DDPM (Denoising Diffusion Probabilistic Model)

Notebook: `DDPM.ipynb`

This notebook implements a diffusion model trained on the **MNIST dataset** to generate handwritten digits.

Main features:

- U-Net architecture for noise prediction  
- Sinusoidal timestep embeddings  
- Exponential Moving Average (EMA) for stable sampling  
- Visualization of forward and reverse diffusion  

The model learns to generate digits by **starting from random noise and gradually denoising it**.

---