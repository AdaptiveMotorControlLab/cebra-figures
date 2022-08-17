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

# # Figure 1: CEBRA for consistent and interpretable embeddings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_hdf("../data/Figure1.h5", key="data")

synthetic_viz = data["synthetic_viz"]
synthetic_scores = {
    k: data["synthetic_scores"][k] for k in ["cebra", "pivae", "umap", "tsne"]
}
viz = data["visualization"]

# ## Figure 1b (left)

plt.figure(figsize=(10, 5))
ax1 = plt.subplot(121)
ax1.scatter(
    synthetic_viz["true"][:, 0],
    synthetic_viz["true"][:, 1],
    s=1,
    c=synthetic_viz["label"],
    cmap="cool",
)
ax1.axis("off")
ax1.set_title("True latent", fontsize=20)
ax2 = plt.subplot(122)
ax2.axis("off")
ax2.set_title("Reconstructed latent", fontsize=20)
ax2.scatter(
    synthetic_viz["cebra"][:, 0],
    synthetic_viz["cebra"][:, 1],
    s=1,
    c=synthetic_viz["label"],
    cmap="cool",
)

# ## Figure 1b (right)

# +
plt.figure(figsize=(5, 5))
ax = plt.subplot(111)

df = pd.DataFrame(synthetic_scores)
sns.stripplot(data=df * 100, color="black", s=3, zorder=1, jitter=0.15)
sns.scatterplot(data=df.median() * 100, color="orange", s=50)
plt.xticks([0, 1, 2, 3], list(synthetic_scores.keys()), fontsize=20, rotation=45)
plt.ylabel("$R^2$", fontsize=20)
plt.yticks(
    np.linspace(0, 100, 11, dtype=int), np.linspace(0, 100, 11, dtype=int), fontsize=20
)
plt.ylim(70, 100)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(axis="both", which="major", labelsize=15)
sns.despine(
    left=False,
    right=True,
    bottom=False,
    top=True,
    trim=True,
    offset={"bottom": 40, "left": 15},
)
# -

# ## Figure 1d

# +
label = viz["label"]
r_ind = label[:, 1] == 1
l_ind = label[:, 2] == 1

fig = plt.figure(figsize=(30, 5))
for i, k in enumerate(["cebra", "cebra_time", "pivae_w", "pivae_wo", "umap", "tsne"]):

    fs = viz[k]
    if not "cebra" in k:
        ax = plt.subplot(1, 6, i + 1)
        ax.scatter(fs[r_ind, 1], fs[r_ind, 0], c=label[r_ind, 0], cmap="viridis", s=1)
        ax.scatter(fs[l_ind, 1], fs[l_ind, 0], c=label[l_ind, 0], cmap="cool", s=1)
        ax.axis("off")
    else:
        ax = plt.subplot(1, 6, i + 1, projection="3d")
        ax.scatter(
            fs[l_ind, 0],
            fs[l_ind, 1],
            fs[l_ind, 2],
            c=label[l_ind, 0],
            cmap="cool",
            s=1,
        )
        ax.scatter(
            fs[r_ind, 0],
            fs[r_ind, 1],
            fs[r_ind, 2],
            c=label[r_ind, 0],
            cmap="viridis",
            s=1,
        )
        ax.axis("off")
    ax.set_title(f"{k}", fontsize=30)
# -

# ## Figure 1e

# +
"""Confusion matrices (Figure 1e).

For all algorithm, evaluate the linear consistency between embeddings computed on the
four different rats.
"""

import pprint

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
import numpy as np
import pathlib

ROOT = pathlib.Path("../data")


def recover_python_datatypes(element):
    if isinstance(element, str):
        if element.startswith("[") and element.endswith("]"):
            if "," in element:
                element = np.fromstring(element[1:-1], dtype=float, sep=",")
            else:
                element = np.fromstring(element[1:-1], dtype=float, sep=" ")
    return element


def load_results(result_name):
    """Load a result file.

    The first line in the result files specify the index columns,
    the following lines are a CSV formatted file containing the
    numerical results.
    """
    results = {}
    for result_csv in (ROOT / result_name).glob("*.csv"):
        with open(result_csv) as fh:
            index_names = fh.readline().strip().split(",")
            df = pd.read_csv(fh).set_index(index_names)
            df = df.applymap(recover_python_datatypes)
            results[result_csv.stem] = df
    return results


results = load_results(result_name="results_v4")
len(results)

plt.rcParams["text.usetex"] = False


def to_cfm(values):
    values = np.concatenate(values)
    assert len(values) == 12, len(values)
    c = np.zeros((4, 4))
    c[:] = float("nan")
    c[np.eye(4) == 0] = values
    return c


def plot_confusion_matrices(results_best):
    fig, axs = plt.subplots(
        ncols=10,
        nrows=1,
        figsize=(14, 1.5),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.08]},
        dpi=500,
    )

    last_ax = axs[-1]

    for ax in axs:
        ax.axis("off")

    for ax, (key, log) in zip(axs[:-1], results_best.items()):
        cfm = log.pivot_table(
            "train", index=log.index.names, columns=["animal"], aggfunc="mean"
        ).apply(to_cfm, axis=1)
        (cfm,) = cfm.values

        sns.heatmap(
            data=np.minimum(cfm * 100, 99),
            vmin=20,
            vmax=100,
            xticklabels=[],
            yticklabels=[],
            cmap=sns.color_palette("gray_r", as_cmap=True),
            annot=True,
            annot_kws={"fontsize": 12},
            cbar=True if (ax == axs[-2]) else False,
            cbar_ax=last_ax if (ax == axs[-2]) else None,
            ax=ax,
        )
        ax.set_title(f"{key} ({100*np.nanmean(cfm):.1f})", fontsize=8)
    return fig


plot_confusion_matrices(results)
plt.show()
