# 🦓 CEBRA - Figures and Data

Code and data for reproducing the figures in
[Learnable latent embeddings for joint behavioral and neural analysis (Schneider, Lee and Mathis, 2022)](https://arxiv.org/abs/2204.00673).

This repo only contains plotting functions which can be applied to pre-computed results. Code for reproducing experiments and applying CEBRA
to custom datasets will be available in the [CEBRA github repository](https://github.com/AdaptiveMotorControlLab/CEBRA).

## Quickstart

Make sure you are in a python>=3.8 environment that supports the `pip install` command (e.g., a virtual environment or a conda environment). Install dependencies, then render of all figures using:

```bash
make -j8 all
```

Figures will be placed in `ipynb` format into the `figures/` directory.

## Dependencies

```bash
pip install -r requirements.txt
```

## Repo organization

- ``src``: Jupyter notebooks for reproducing the paper figures, in python format
- ``data``: Folder to data files
- ``figures``: Rendered paper figures in `ipynb` format
