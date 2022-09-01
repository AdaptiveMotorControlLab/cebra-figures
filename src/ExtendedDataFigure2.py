# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Extended Data Figure 2: Hyperparameter changes on visualization and consistency.

# ### Temperature has the largest effect on visualization (vs. consistency) of the embedding
#
# - shown by a range from 0.1 to 3.21 (highest consistency for Rat 1), as can be appreciated in 3D (top) and post FastICA into a 2D embedding (middle). Bottom row shows the corresponding change on mean consistency. 
#
# - Note, tempature is both a learnable and easily modified parameter in CEBRA.

# +
# TODO: Temperature ablation plots
# -

# - Orange line denotes the median and black dots are individual runs (subject consistency: 10 runs with 3 comparisons per rat; run consistency: 10 runs, each compared to 9 remaining runs).

# +
# Consistency comparison across temperature values
