# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Extended Data Figure 3: CEBRA produces consistent, highly decodable embeddings

# - Additional rat data shown for all algorithms we benchmarked (see Methods). CEBRA was trained with output latent 3D (the minimum) and all other methods were obtained with a 2D latent

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scatter(data, index, ax, s=0.01, alpha=0.5):
    mask = index[:, 1] > 0
    ax.scatter(*data[mask].T, c=index[mask, 0], s=s, cmap="viridis", alpha=alpha)
    ax.scatter(*data[~mask].T, c=index[~mask, 0], s=s, cmap="cool", alpha=alpha)


df = pd.concat(
    [
        pd.read_hdf("../data/EDFigure3.h5", key="data"),
        pd.read_hdf("../data/EDFigure3_extension.h5", key="data"),
    ],
    axis=0,
).reset_index(drop=True)

fig = plt.figure(figsize=(4 * 3, 7 * 3), dpi=600)
for i in df.index:
    ax = fig.add_subplot(7, 4, i + 1)
    scatter(df.loc[i, "emission"][:, :2], df.loc[i, "labels"], ax=ax, s=0.5, alpha=0.7)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(bottom=True, left=True, ax=ax)
    # first row labels
    if i // 4 == 0:
        ax.set_title(f"Rat {df.loc[i, 'animal']}", fontsize=18)
    # first column labels
    if i % 4 == 0:
        ax.set_ylabel(df.loc[i, "method"], fontsize=18)
