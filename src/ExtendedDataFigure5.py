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

# # Extended Data Figure 5: Hypothesis testing with CEBRA.

# ### Example data from a hippocampus recording session (Rat 1). 
#
# - We test possible relationships between three experimental variables (rat location, velocity, movement direction) and the neural recordings (120 neurons, not shown).

# +
# TODO:
# -

# ### Relationship between velocity and position.

# +
# TODO:
# -

# ### Training CEBRA with three-dimensional outputs on every single experimental variable (main diagonal) and every combination of two variables. 
#
# - All variables are treated as "continuous" in this experiment. We compare original to shuffled variables (shuffling is done by permuting all samples over the time dimension) as a control. We project the original three dimensional space onto the first principal components. We show the minimum value of the InfoNCE loss on the trained embedding for all combinations in the confusion matrix (lower number is better). Either velocity or direction, paired with position information is needed for maximum structure in the embedding (highlighted, colored), yielding lowest InfoNCE error.

# ### Using an eight-dimensional CEBRA embedding does not qualitatively alter the results. 
#
# - We again report the first two principal components as well as InfoNCE training error upon convergence, and find non-trivial embeddings with lowest training error for combinations of direction/velocity and position.

# +
# TODO:
