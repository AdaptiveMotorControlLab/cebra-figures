import numpy as np
import pandas as pd

def concat(single, multi, time):
  metrics = list(single.columns[:-5])
  print(metrics)
  del metrics[metrics.index('logdir')]

  single['data_mode'] = 'single-session'
  multi['data_mode'] = 'multi-session'
  time['data_mode'] = 'time-contrastive'

  results = pd.concat(
    [
      single.set_index(metrics),
      multi.set_index(metrics),
      time.set_index(metrics)
    ]
  )
  return metrics, results

root = "/home/stes/ssh/cebra_public/results/figure_4/csvs"

single = pd.read_csv(f"{root}/leave2out-single-bsz7200-long.csv", index_col=0)
multi = pd.read_csv(f"{root}/leave2out-multi-bsz7200-long.csv", index_col=0)
time = pd.read_csv(f"{root}/leave2out-single-bsz7200.csv", index_col=0)
metrics, large_batches = concat(single, multi, time)

#single = pd.read_csv("/home/stes/ssh/cebra_public/leave2out.csv", index_col=0)
#multi = pd.read_csv("/home/stes/ssh/cebra_public/leave2out-multi.csv", index_col=0)
#time = pd.read_csv("/home/stes/ssh/cebra_public/leave2out-timecl.csv", index_col=0)
#metrics, small_batches = concat(single, multi, time)

results = pd.concat([
  large_batches,
  #small_batches  
]).reset_index()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def figure():
  plt.figure(figsize = (3, 2), dpi = 200)


def avg(results, maps):
  results = results.copy()
  for key, remapped in maps.items():
    remapped = key if remapped is None else remapped
    results[remapped + '_raw'] = results[key].apply(lambda v : np.array(eval(v)))
    results[remapped] = results[key].apply(lambda v : np.array(eval(v)).mean())
    
    if key != remapped:
      del results[key]
  return results

maps = {
  'train' : "train_consistency",
  'valid' : "valid_consistency",
  'test' : "test_consistency",
  'valid_accuracy' : None,
  'test_accuracy' : None
}

def agg(values):
  if values.name.endswith("_raw"):
    return np.stack(values, axis = 0).mean(axis = 0)
  return values.mean()

results = avg(results, maps)
metrics = [maps.get(m, m) for m in metrics]
results = results.groupby([ c for c in metrics + ["data_mode"] if c != 'repeat']).agg(agg).reset_index()

print(results.columns)