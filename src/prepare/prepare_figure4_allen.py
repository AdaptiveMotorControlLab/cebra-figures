import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def concat(single, multi):
    metrics = list(single.columns[:-5])
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


def avg(results, maps):
    results = results.copy()
    for key, remapped in maps.items():
        remapped = key if remapped is None else remapped
        results[remapped + "_raw"] = results[key].apply(lambda v: np.array(eval(v)))
        results[remapped] = results[key].apply(lambda v: np.array(eval(v)).mean())

        if key != remapped:
            del results[key]
    return results


def agg(values):
    if values.name.endswith("_raw"):
        return np.stack(values, axis=0).mean(axis=0)
    return values.mean()


if __name__ == "__main__":

    maps = {
        "train": "train_consistency",
        "valid": "valid_consistency",
        "test": "test_consistency",
        "valid_accuracy": None,
        "test_accuracy": None,
    }

    root = "./ssh/cebra_public/results/figure_4/csvs"

    single = pd.read_csv(f"{root}/leave2out-single-bsz7200-long.csv", index_col=0)
    multi = pd.read_csv(f"{root}/leave2out-multi-bsz7200-long.csv", index_col=0)
    metrics, large_batches = concat(single, multi)
    results = large_batches.reset_index()

    results = avg(results, maps)
    metrics = [maps.get(m, m) for m in metrics]
    results = (
        results.groupby([c for c in metrics + ["data_mode"] if c != "repeat"])
        .agg(agg)
        .reset_index()
    )

    results.to_csv("../data/Figure4SupplementMultisession.csv")
