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

# # Table S1

# +
import scipy.stats
import numpy as np
import joblib as jl
import sklearn.metrics
import pandas as pd

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import get_tukey_pvalue
# -

DATA = '../data/SupplTable1.h5'
supervised_methods = ['cebra_10_b', 'pivae_1_w', 'pivae_10_w', 'pivae_1_wo', 'pivae_10_wo']
supervised_methods_decoding = ['cebra_10_b', 'pivae_1_mcmc', 'pivae_10_mcmc', 'pivae_1_wo', 'pivae_10_wo']
unsupervised_methods = ['cebra_10_t', 'tsne', 'umap', 'pca']


# +
# for decoding
# avg over seeds
#    (# animals x # of CV runs) --> 4 x 3 --> 12

def decoding(key, animal = 0):
  data = pd.read_hdf(DATA, key = key)
  metric = "test_position_error"
  if metric + "_svm" in data.columns:
    metric =  metric + "_svm"
  data = data.pivot_table(
    metric,
    index = "animal",
    columns = "seed",
    aggfunc = lambda v : np.mean(np.array(v))
  ).agg(np.array, axis = 1)
  
  if animal is None:
    return data.agg(np.concatenate, axis = 0)
  else:
    return data.loc[animal]


for animal in [0]:
  
  def load_data(keys):
    return pd.DataFrame([
      {"method" : key, "metric" : decoding(key, animal)}
      for key in keys
    ])

  # Supervised
  data = load_data(supervised_methods_decoding)
  anova_sup = scipy.stats.f_oneway(*data.metric.values)
  data_explode = data.explode("metric")
  data_explode.metric = data_explode.metric.astype(float)
  posthoc_sup = pairwise_tukeyhsd(
    data_explode.metric.values,
    data_explode.method.values,
    alpha = 0.05
  )
  sup_data = data
  
  # Unsupervised
  data = load_data(unsupervised_methods)
  data_explode = data.explode("metric")
  print(len(data_explode))
  data_explode.metric = data_explode.metric.astype(float)
  anova = scipy.stats.f_oneway(*data.metric.values)
  posthoc = pairwise_tukeyhsd(
    data_explode.metric.values,
    data_explode.method.values,
    alpha = 0.05
  )
  unsup_data = data
  
  import seaborn as sns
  import matplotlib.pyplot as plt

  print(f"""
  # Decoding (animal filter: {animal})


  ## Supervised Methods
  
  Anova:  {anova_sup}

  Post Hoc test:

  {posthoc_sup}
  p-values: {posthoc_sup.pvalues}
  
  ## Unsupervised Methods

  Anova:  {anova}

  Post Hoc test:

  {posthoc}
  p-values: {posthoc.pvalues}
  """
  )
  
  fig, axes = plt.subplots(1,2, figsize = (8, 3))
  sns.boxplot(data = sup_data.explode("metric"), x = "method", y = "metric", ax = axes[0])  
  sns.boxplot(data = unsup_data.explode("metric"), x = "method", y = "metric", ax = axes[1])
  axes[0].set_title("Supervised")
  axes[1].set_title("Unsupervised")
  for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
  plt.tight_layout()
  plt.show()


# +
def subject_consistency(key):
  if key == "autolfads":
    autolfads_consistency = np.array(
    [[           0.52405768, 0.54354575, 0.5984262 ],
     [0.61116595,            0.59024053, 0.747014  ],
     [0.68505602, 0.60948229,            0.57858312],
     [0.77841349, 0.78809085, 0.65031025          ]])

    return autolfads_consistency.flatten()
  else:
    data =  pd.read_hdf(DATA, key = key).pivot_table(
      "train",
      columns = "animal",
      aggfunc = lambda v : np.mean(np.array(v))
    ).agg(np.concatenate, axis = 1).item()
    return data

def load_data(keys):
  return pd.DataFrame([
    {"method" : key, "metric" : subject_consistency(key)}
    for key in keys
  ])

methods = ['cebra_10_b', 'cebra_10_t', 'pivae_10_w', 'pivae_10_wo', 'tsne', 'umap', 'autolfads']

# Supervised
data = load_data(methods)
anova_sup = scipy.stats.f_oneway(*data.metric.values)
data_explode = data.explode("metric")
data_explode.metric = data_explode.metric.astype(float)
posthoc_sup = pairwise_tukeyhsd(
  data_explode.metric.values,
  data_explode.method.values,
  alpha = 0.05
)

print(f"""
# Subject Consistency

Anova:  {anova_sup}

Post Hoc test:

{posthoc_sup}
p-values: {posthoc_sup.pvalues}
"""
)

fig, ax = plt.subplots(1,1, figsize = (8, 3))
sns.boxplot(data = data.explode("metric"), x = "method", y = "metric", ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
plt.show()
# -

rat = 0
print(f"Rat {rat}")
for key in supervised_methods_decoding:
  print(f"\t{key}\t{decoding(key, animal=rat).mean():.5f} +/- {decoding(key, animal=rat).std():.5f}")
