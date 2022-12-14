# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# # Table S1-2

import joblib as jl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.metrics
from statsmodels.sandbox.stats.multicomp import get_tukey_pvalue
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway


def anova_with_report(data):
    # One way ANOVA, helper function for formatting
    control = scipy.stats.f_oneway(*data)
    print(control)
    a = anova_oneway(
        data,
        use_var="equal",
    )
    assert np.isclose(a.pvalue, control.pvalue), (a.pvalue, control.pvalue)
    assert np.isclose(a.statistic, control.statistic)
    return f"F = {a.statistic}, p = {a.pvalue}\n\n    " + "\n    ".join(
        str(a).split("\n")
    )


DATA = "../data/SupplTable1.h5"

# ## Table S1: Consistency across subjects

# We compare the consistency across subjects of all available methods depicted in Figure 1:

methods = [
    "cebra_10_b",
    "cebra_10_t",
    "pivae_10_w",
    "pivae_10_wo",
    "tsne",
    "umap",
    "autolfads",
]


# +
def subject_consistency(key):
    if key == "autolfads":
        autolfads_consistency = np.array(
            [
                [0.52405768, 0.54354575, 0.5984262],
                [0.61116595, 0.59024053, 0.747014],
                [0.68505602, 0.60948229, 0.57858312],
                [0.77841349, 0.78809085, 0.65031025],
            ]
        )
        return autolfads_consistency.flatten()
    else:
        data = (
            pd.read_hdf(DATA, key=key)
            .pivot_table(
                "train", columns="animal", aggfunc=lambda v: np.mean(np.array(v))
            )
            .agg(np.concatenate, axis=1)
            .item()
        )
        return data


def load_data(keys):
    return pd.DataFrame(
        [{"method": key, "metric": subject_consistency(key)} for key in keys]
    )


# +
data = load_data(methods)
anova_sup = scipy.stats.f_oneway(*data.metric.values)
data_explode = data.explode("metric")
data_explode.metric = data_explode.metric.astype(float)
data_explode.sort_values("metric")
posthoc_sup = pairwise_tukeyhsd(
    data_explode.metric.values, data_explode.method.values, alpha=0.05
)

print(
    f"""
# Subject Consistency

Anova:  {anova_sup}

Post Hoc test:

{posthoc_sup}
p-values: {posthoc_sup.pvalues}
"""
)

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
sns.boxplot(data=data.explode("metric"), x="method", y="metric", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
# -

# ## Table S2: Decoding performance

# We compare the methods trained using label information (CEBRA-behavior and all considered variants of piVAE) and the self-/un-supervised methods trained without using labels (CEBRA-time, t-SNE, UMAP, autoLFADS, and the PCA baseline).

supervised_methods = [
    "cebra_10_b",
    "pivae_1_w",
    "pivae_10_w",
    "pivae_1_wo",
    "pivae_10_wo",
]
supervised_methods_decoding = [
    "cebra_10_b",
    "pivae_1_mcmc",
    "pivae_10_mcmc",
    "pivae_1_wo",
    "pivae_10_wo",
]
unsupervised_methods = ["cebra_10_t", "tsne", "umap", "autolfads", "pca"]


# +
# for decoding
# avg over seeds
#    (# animals x # of CV runs) --> 4 x 3 --> 12


def decoding(key, animal=0):
    data = pd.read_hdf(DATA, key=key)
    metric = "test_position_error"
    if metric + "_svm" in data.columns:
        metric = metric + "_svm"
    data = data.pivot_table(
        metric, index="animal", columns="seed", aggfunc=lambda v: np.mean(np.array(v))
    ).agg(np.array, axis=1)

    if animal is None:
        return data.agg(np.concatenate, axis=0)
    else:
        return data.loc[animal]


def load_data(keys, animal):
    return pd.DataFrame(
        [{"method": key, "metric": decoding(key, animal)} for key in keys]
    ).copy()


def report_supervised(animal):
    data = load_data(supervised_methods_decoding, animal)
    anova = anova_with_report(data.metric.values)
    data_explode = data.explode("metric")
    data_explode.metric = data_explode.metric.astype(float)
    posthoc_sup = pairwise_tukeyhsd(
        data_explode.metric.values, data_explode.method.values, alpha=0.05
    )
    return anova, posthoc_sup, data


def report_unsupervised(animal):
    data = load_data(unsupervised_methods, animal)
    data.loc[(data["method"] == "pca"), "metric"] = data[(data["method"] == "pca")][
        "metric"
    ].apply(lambda v: v.repeat(10))
    data_explode = data.explode("metric")
    data_explode.metric = data_explode.metric.astype(float)
    anova = anova_with_report(data.metric.values)
    posthoc = pairwise_tukeyhsd(
        data_explode.metric.values, data_explode.method.values, alpha=0.05
    )
    return anova, posthoc_sup, data


def plot_overview(sup_data, unsup_data):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    sns.boxplot(data=sup_data.explode("metric"), x="method", y="metric", ax=axes[0])
    sns.boxplot(data=unsup_data.explode("metric"), x="method", y="metric", ax=axes[1])
    axes[0].set_title("Supervised")
    axes[1].set_title("Unsupervised")
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()


# -

# ### Rat 1 (Achilles), Anova and post-hoc test on supervised methods

# +
anova, posthoc, sup_data = report_supervised(0)

print(anova)
print("\n\n")
print(posthoc)
print("p-values: ", ", ".join(map(str, posthoc.pvalues)))
# -

# ### Rat 1 (Achilles), Anova and post-hoc test on unsupervised methods

# +
anova, posthoc, unsup_data = report_unsupervised(0)

print(anova)
print("\n\n")
print(posthoc)
print("p-values: ", ", ".join(map(str, posthoc.pvalues)))
# -

# ### Rat 1 (Achilles), overview plot
#
# (Not shown in the paper)

plot_overview(sup_data, unsup_data)

# ### All Rats, Anova and post-hoc test on supervised methods

# +
anova, posthoc, sup_data = report_supervised(None)

print(anova)
print("\n\n")
print(posthoc)
print("p-values: ", ", ".join(map(str, posthoc.pvalues)))
# -

# ### All Rats, Anova and post-hoc test on unsupervised methods

# +
anova, posthoc, unsup_data = report_unsupervised(None)

print(anova)
print("\n\n")
print(posthoc)
print("p-values: ", ", ".join(map(str, posthoc.pvalues)))
# -

plot_overview(sup_data, unsup_data)

# ### Overview of the decoding performance

rat = 0
print(f"Rat {rat}")
for key in supervised_methods_decoding:
    print(
        f"\t{key}\t{decoding(key, animal=rat).mean():.5f} +/- {decoding(key, animal=rat).std():.5f}"
    )
