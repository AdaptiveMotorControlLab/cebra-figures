# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import scipy.stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import get_tukey_pvalue
import joblib
import pandas as pd

decoding_results=pd.read_hdf('../data/Figure5Revision.h5', data='key')


# +
def get_data(decoding_results, task, decoders, methods, window, modality, num_neurons):
    if modality == 'ca':
        index = 0
    elif modality == 'np':
        index = 1
    accs=[]
    keys=[]
    for decoder, method in zip(decoders,methods):
        key = f'{modality}_{method}_{window}'
        if 'joint' in method:
            seeds = decoding_results[task][decoder][key][num_neurons]
            acc= [abs(s[index]).mean() for s in seeds]
        else:
            acc= abs(np.array(decoding_results[task][decoder][key][num_neurons])).mean(axis=-1)
        accs.append(acc)
        keys.append([f'{key}_{decoder}']*len(acc))
    return np.concatenate(accs),np.concatenate(keys)

    
def concat_neurons(decoding_results, task, decoder, method, window, modality, n = 1000):
    if modality == 'ca':
        index = 0
    elif modality == 'np':
        index = 1
    key = f'{modality}_{method}_{window}'
    accs=[]
    if 'joint' in method:
        seeds = decoding_results[task][decoder][key][n]
        accs.append([abs(s[index]).mean() for s in seeds])
    else:
        accs.append(abs(np.array(decoding_results[task][decoder][key][n])).mean(axis=-1))
    return np.concatenate(accs)


# -

# ## ANOVA for CEBRA, CEBRA-joint, baseline 330 ms (10 frame window), 1000 neurons:

# +
np_total_stats = scipy.stats.f_oneway(concat_neurons(decoding_results, 'frame_err', 'knn', 'cebra', '330', 'np'), 
                                      concat_neurons(decoding_results, 'frame_err', 'knn', 'cebra_joint', '330', 'np'), 
                                      concat_neurons(decoding_results, 'frame_err', 'knn', 'baseline', '330', 'np'),
                                     concat_neurons(decoding_results, 'frame_err', 'bayes', 'baseline', '330', 'np'))



print(f'NP total stats \n {np_total_stats}')
# -

# ## ANOVA for CEBRA, CEBRA-joint, baseline for each neuron numbers

num_neurons = [1000]
for i in num_neurons:
    print(f'For {i} neurons from np recording (330ms):')
    
    np_data, np_keys = get_data(decoding_results, 'frame_err', ['knn', 'knn', 'knn', 'bayes'], ['cebra', 'cebra_joint', 'baseline', 'baseline'], '330', 'np', i)
    
    stats=pairwise_tukeyhsd(np_data.flatten(), np_keys,)
    print(stats)
