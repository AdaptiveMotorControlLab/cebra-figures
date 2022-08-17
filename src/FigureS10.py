# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Figure S10: CEBRA produces consistent, highly decodable embeddings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model

data = pd.read_hdf("../data/FigureS10.h5", key="data")

fig = plt.figure(figsize=(20, 24))
emissions_list = [
    data["cebra"],
    data["emission_cebra_time"],
    data["emission_umap"],
    data["emission_tsne"],
    data["emission_pivae_w"],
    data["emission_pivae_wo"],
]
for i in range(4):
    for j in range(6):

        ax = fig.add_subplot(6, 4, j * 4 + i + 1)
        idx1, idx2 = (0, 1)
        lin = sklearn.linear_model.LinearRegression()
        lin.fit(emissions_list[j][i], emissions_list[j][0])
        fitted = lin.predict(emissions_list[j][i])

        ax.scatter(
            fitted[:, idx1],
            fitted[:, idx2],
            c=np.tile(np.arange(900), 10),
            cmap="magma",
            s=1,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            plt.title(["Mouse 1", "Mouse 2", "Mouse 3", "Mouse 4"][i], fontsize=25)
        if i == 0:
            plt.ylabel(
                [
                    "CEBRA-Behavior",
                    "CEBRA-Time",
                    "UMAP",
                    "TSNE",
                    "pi-VAE w/ label",
                    "pi-VAE w/o label",
                ][j],
                fontsize=25,
            )
