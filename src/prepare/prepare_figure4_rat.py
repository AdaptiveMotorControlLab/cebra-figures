"""Helper utils for the supplement figure on multisession training.

This utility script is for the hippocampus dataset.

For http://localhost:8888/notebooks/code/cebra/neural_cl/cebra_public/results/Figure4_MultiSession_Supplement.ipynb
"""

import pandas as pd
import numpy as np
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(fname):
    """Read dataset from the given (server) filename."""

    with open(fname, "r") as fh:
        single = filter(len, fh.read().split("@@"))

    data = []
    errors = []
    for i, line in enumerate(single):
        try:
            line = json.loads(line)
        except json.JSONDecodeError as e:
            try:
                first, second = line.split("}{")
                first = json.loads(first + "}")
                second = json.loads("{" + second)
                assert first == second
                line = first
            except:
                errors.append(i)
                continue
        data.append(
            {
                key: (tuple(value) if isinstance(value, list) else value)
                for key, value in line.items()
            }
        )
    print(len(errors))
    return pd.DataFrame(data), errors


# single, single_errors = read_data('/home/stes/ssh/cebra_public/results/figure_4/hippocampus_hiddendims/singlesession.json')
# multi, multi_errors = read_data('/home/stes/ssh/cebra_public/results/figure_4/hippocampus_hiddendims/multisession.json')


def concat(single, multi):
    metrics = list(single.columns[:-10])
    print(metrics)
    del metrics[metrics.index("logdir")]

    single["data_mode"] = "single-session"
    multi["data_mode"] = "multi-session"

    results = pd.concat(
        [
            single.set_index(metrics),
            multi.set_index(metrics),
        ]
    )
    return metrics, results


def select_rat_1(line):
    """Aggregation function (for groupby, or pivot_table) for selecting the first rat."""
    if len(line) == 4:
        return line[0]
    if len(line) == 12:
        return np.mean(line[0:3])
    return line


def avg(results, maps):
    """Aggregation function to compute the average results for each key in the mapping."""
    results = results.copy()
    for key, remapped in maps.items():
        remapped = key if remapped is None else remapped
        results[remapped + "_raw"] = results[key].apply(lambda v: np.array(v))
        results[remapped] = results[key].apply(
            np.mean
        )  # .apply(select_rat_1) # .mean()
        if key != remapped:
            del results[key]
    return results


def agg(values):
    if values.name.endswith("_raw"):
        return np.stack(values, axis=0).mean(axis=0)
    return values.mean()


def load_data():
    maps = {
        "train": "train_consistency",
        "valid": "valid_consistency",
        "test": "test_consistency",
        "valid_total_r2": None,
        "test_total_r2": None,
        "valid_position_error": None,
        "test_position_error": None,
    }

    single, single_errors = read_data(
        "./ssh/cebra_public/results/figure_4/hippocampus_hiddendims/singlesession.json"
    )
    multi, multi_errors = read_data(
        "./ssh/cebra_public/results/figure_4/hippocampus_hiddendims/multisession.json"
    )
    metrics, results = concat(single, multi)
    results = results.reset_index()

    results = avg(results, maps)

    print(metrics)
    print(results)
    results.to_csv("../data/Figure4SupplementMultisession_rat.csv")


load_data()
