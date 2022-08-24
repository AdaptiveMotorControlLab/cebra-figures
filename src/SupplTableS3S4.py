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

# # Table S3-4

import scipy.stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import get_tukey_pvalue
import joblib
import pandas as pd

decoding_results=pd.read_hdf('../data/Figure5Revision.h5', data='key')


# +
def get_data2(decoding_results, task, decoders, methods, window, modality, num_neurons):
    if modality == 'ca':
        index = 0
    elif modality == 'np':
        index = 1
    accs=[]
    keys=[]
    for decoder, method in zip(decoders,methods):
        key = f'{modality}_{method}_{window}'
        if 'joint' in method:
            acc =np.array(decoding_results[task][decoder][key][num_neurons])[:, index]
        else:
            acc=np.array(decoding_results[task][decoder][key][num_neurons])
        accs.append(acc)
        keys.append([f'{key}_{decoder}']*len(acc))
    return np.concatenate(accs),np.concatenate(keys)

    
def concat_neurons(decoding_results, task, decoder, method, window, modality):
    if modality == 'ca':
        index = 0
    elif modality == 'np':
        index = 1
    key = f'{modality}_{method}_{window}'
    accs=[]
    for n in decoding_results[task][decoder][key].keys():
        if 'joint' in method:
            accs.append(np.array(decoding_results[task][decoder][key][n])[:, index])
        else:
            accs.append(np.array(decoding_results[task][decoder][key][n]))
    return np.concatenate(accs)


# -

# ### ANOVA for CEBRA, CEBRA-joint, baseline 330 ms (10 frame window):

# +
np_total_stats = scipy.stats.f_oneway(concat_neurons(decoding_results, 'frame_id', 'knn', 'cebra', '330', 'np'), 
                                      concat_neurons(decoding_results, 'frame_id', 'knn', 'cebra_joint', '330', 'np'), 
                                      concat_neurons(decoding_results, 'frame_id', 'knn', 'baseline', '330', 'np'),
                                     concat_neurons(decoding_results, 'frame_id', 'bayes', 'baseline', '330', 'np'))



print(f'NP total stats \n {np_total_stats}')
# -

# ### ANOVA for CEBRA, CEBRA-joint, baseline 33 ms (1 frame window):

# +
np_total_stats = scipy.stats.f_oneway(concat_neurons(decoding_results, 'frame_id', 'knn', 'cebra', '33', 'np'), 
                                      concat_neurons(decoding_results, 'frame_id', 'knn', 'cebra_joint', '33', 'np'), 
                                      concat_neurons(decoding_results, 'frame_id', 'knn', 'baseline', '33', 'np'),
                                     concat_neurons(decoding_results, 'frame_id', 'bayes', 'baseline', '33', 'np'))



print(f'NP total stats \n {np_total_stats}')
# -

# ### ANOVA for CEBRA, CEBRA-joint, baseline for each neuron numbers

num_neurons = [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000]
for i in num_neurons:
    print(f'For {i} neurons from np recording (330ms):')
    
    np_data, np_keys = get_data2(decoding_results, 'frame_id', ['knn', 'knn', 'knn', 'bayes'], ['cebra', 'cebra_joint', 'baseline', 'baseline'], '330', 'np', i)
    
    stats=pairwise_tukeyhsd(np_data.flatten(), np_keys,)
    print(stats)

num_neurons = [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000]
for i in num_neurons:
    print(f'For {i} neurons from np recording (33ms):')
    
    np_data, np_keys = get_data2(decoding_results, 'frame_id', ['knn', 'knn', 'knn', 'bayes'], ['cebra', 'cebra_joint', 'baseline', 'baseline'], '33', 'np', i)
    
    stats=pairwise_tukeyhsd(np_data.flatten(), np_keys)
    print(stats)


