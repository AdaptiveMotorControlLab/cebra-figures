"""Helper functions for loading and parsing logs for Figure 2 results."""

import glob
import json
import pandas as pd
import numpy as np
import joblib
import pathlib

ROOT = pathlib.Path("./ssh/cebra_public/results")

def load_result(name: str, pattern: str, num_seeds: int, root: pathlib.Path = ROOT):

    metrics = [
        "all_run_consistency",
        "train_run_consistency",
        "valid_run_consistency",
        "test_run_consistency",
        "all",
        "train",
        "valid",
        "test",
        "knn",
        "valid_total_r2",
        "valid_position_error",
        "valid_position_r2",
        "test_total_r2",
        "test_position_error",
        "test_position_r2",
        "test_total_r2_svm",
        "test_position_error_svm",
        "test_position_r2_svm",
    ]

    def _load_dfs(files):
        dfs = []
        for fname in sorted(files):
            with open(fname, "r") as fh:
                data = list(map(json.loads, filter(len, fh.read().split("\n"))))
            df = pd.DataFrame(data)
            dfs.append(df.drop(columns="group_by"))
        assert len(dfs) == 2, pattern
        return dfs

    def _parse_shape(shape):
        if len(shape) == 1:
            return 1, shape[0]
        elif len(shape) == 2:
            return shape
        raise ValueError(shape)

    by_animal, by_seed = _load_dfs(root.glob(pattern))
    (shape,) = by_animal.train_run_consistency.apply(
        lambda v: np.array(v).shape
    ).unique()
    num_animals, cfm_size = _parse_shape(shape)

    params = list(set(by_animal.columns).difference(metrics))

    # TODO(stes)
    if num_animals == 1:
        by_animal["animal"] = np.tile(np.arange(0, 4), len(by_animal) // 4)
        params_wo_animal = list(set(params).difference({"animal"}))
        by_animal = by_animal.groupby(params_wo_animal).agg(list).reset_index()
        num_animals = 4

    # return by_animal, by_seed
    cfm_size_animals = by_seed.valid.apply(len).unique()
    assert cfm_size == num_seeds * (num_seeds - 1)
    assert cfm_size_animals == 4 * (4 - 1)

    def add_values(key, num):
        def _add_values(df):
            df[key] = np.arange(num)
            return df

        return _add_values

    # expand the values in the results that were collected for each repeat
    # of the experiment. For the consistency metrics, we get vectors of shape (12,)
    # and of shape (4,) for decoding metrics. The seed is represented as a non-unique
    # index.
    by_seed = by_seed.groupby(params).apply(add_values("seed", num_seeds))
    metric_names = list(set(by_seed.columns).intersection(metrics))
    c_metrics = ["train", "valid", "test", "all"]
    for m in c_metrics:
        if m in by_seed.columns:
            by_seed[m] = by_seed[m].apply(lambda v: np.array(v).reshape(4, 3))
    by_seed = (
        by_seed.explode(metric_names)
        .groupby(params + ["seed"])
        .apply(add_values("animal", 4))
    )

    # expand the values in the results that were collected for each animal, across
    # repeats of the experiment. We get vectors of shape (4, N(N-1),) depending on the
    # number N of seeds.
    metric_names = list(set(by_animal.columns).intersection(metrics))
    for key in metric_names:
        by_animal[key] = by_animal[key].apply(
            lambda v: np.array(v).reshape(4, num_seeds, num_seeds - 1).tolist()
        )
    by_animal = (
        by_animal.explode(metric_names)
        .groupby(params)
        .apply(add_values("animal", 4))
        .explode(metric_names)
        .groupby(params + ["animal"])
        .apply(add_values("seed", num_seeds))
    )

    assert len(by_seed) == len(by_animal)

    by_seed_indexed = by_seed.set_index(["animal", "seed"] + params)
    by_animal_indexed = by_animal.set_index(["animal", "seed"] + params)
    aggregated = pd.concat([by_seed_indexed, by_animal_indexed], axis=1)
    aggregated.columns.name = name
    return aggregated


def _load(*args, **kwargs):
    try:
        return load_result(*args, **kwargs)
    except Exception as e:
        print("[FAIL]", args, kwargs)
        raise e


def compute_best(frame, metric, temp_threshold=None, head=True):
    """Compute best set of parameters using cross-validation."""

    animal_frame = frame.copy()  #
    # animal_frame = frame[frame.index.get_level_values('animal') == 0].copy()

    agg_metric = f"{metric}_agg"
    params = animal_frame.index.names[2:]
    animal_frame[agg_metric] = animal_frame[metric].apply(lambda v: np.array(v).mean())
    group = set(params)
    group.remove("repeat")
    group = list(group)
    # temp_threshold

    best = animal_frame.groupby(group)[agg_metric].mean().sort_values(ascending=False)
    if temp_threshold is not None:
        index_names = best.index.names
        best = best.reset_index()
        idc = best.temperature
        idc = (idc > temp_threshold[0]) & (idc < temp_threshold[1])
        best = best[idc].set_index(index_names)

    #print()
    #display(best.head(3).to_frame())

    if head:
        best = best.head(1)
    else:
        best = best.tail(1)

    # best_ = best.reset_index()
    # print(len(best_))
    # display(best_.head(5))
    # display(best_[best_.temperature < 1.5].head(5))

    return frame.reset_index().set_index(group).loc[best.index]


def select_results(results):
    """Select best results by cross-validation."""
    results_best = {}
    params = {}
    for key, value in results.items():
        # if key != "cebra-t-sweep": continue
        #print(key)
        # if key == "cebra-1-dims":
        #  idc = results[key].
        if key == "cebra-t-lowertemp":
            for temp_threshold in [(0.05, 0.15), (0.15, 0.25), (1, 2)]:
                results_best[f"cebra-t_{temp_threshold[1]}"] = compute_best(
                    value, "valid_test_error", temp_threshold
                )
        else:
            if "dim" in key:
                for dim in (8, 16, 32):
                    level = value.index.names.index("output_dimension")
                    results_best[f"{dim}d_{key}"] = compute_best(
                        value.xs(dim, level=level), "train", None, head=True
                    )
            else:
                results_best[key] = compute_best(value, "train", None, head=True)
            # idx = results_best[key].index
            # params[key] = dict(zip(idx.names, idx[0]))
    return results_best

def main(results_fn):
    results = results_fn()
    results_best = select_results(results)
    for key, filename in results_best.items():
        root = '../data' / pathlib.Path(results_fn.__name__)
        root.mkdir(exist_ok=True)
        output = root / f'{key}.csv'
        with output.open('w') as fh:
            index_names = ','.join(filename.index.names)
            print(index_names, file = fh)
        filename.to_csv(root / f'{key}.csv', mode = 'a')
        print("Writing", root / f'{key}.csv')

def results_v1():
    return {
        key: _load(key, fname, num_seeds=num_seeds)
        for key, fname, num_seeds in [
            # ("cebra-b-defaults", f"{root}/figure2_update_v3/cebra-behaviour-defaults_*.json", 10),
            (
                "cebra-b-sweep",
                "figure2_update_v4/figure2-cebra-behavior-sweep-grid-2_*.json",
                3,
            ),
            # ("cebra-b-rat0", f"{root}/figure2_update_v5/cebra-behavior_*.json", 10),
            ("cebra-b", f"figure2_update_v5/cebra-behavior-allrats*.json", 10),
            (
                "cebra-b-s1",
                "figure2_update_v5/cebra-behavior-step1-sweep*.json",
                3,
            ),
            ("pivae-w", f"figure2_update_v5/pivae-with-labels_*.json", 10),
            # ("cebra-t-defaults", f"{root}/ /cebra-time-defaults_*.json", 10),
            (
                "cebra-t-sweep",
                f"figure2_update_v4/figure2-cebra-time-sweep-grid-2_*.json",
                3,
            ),
            # ("cebra-t-rat0", f"{root}/figure2_update_v5/cebra-time_*.json", 10),
            # ("cebra-t-lowertemp", f"{root}/figure2_update_v5/cebra-time-lowertemp*.json", 10),
            ("cebra-t", "figure2_update_v5/cebra-time-allrats*.json", 10),
            ("cebra-t-s1", f"figure2_update_v5/cebra-time-step1-sweep*.json", 3),
            ("pivae-wo", "figure2_update_v3/pivae-without-labels_*.json", 10),
            # ("tsne", f"{root}/figure2_update_v3/tsne*grid-v2*.json", 3),
            # ("tsne-rat0", f"{root}/figure2_update_v5/tsne_*", 10),
            ("tsne", f"figure2_update_v5/tsne-allrats*", 10),
            # ("umap", f"{root}/figure2_update_v3/umap*grid-v2*.json", 3),
            # ("umap-rat0", f"{root}/figure2_update_v5/umap_*", 10),
            ("umap", f"figure2_update_v5/umap-allrats*", 10),
            ("pca", f"figure2_update_v5/pca*", 1),
        ]
    }


def results_v2():
    return {
        key: _load(key, fname, num_seeds=num_seeds)
        for key, fname, num_seeds in [
            ("cebra-b", f"figure2_update_v6/cebra-behavior-allrats*.json", 10),
            ("pivae-w", f"figure2_update_v5/pivae-with-labels_*.json", 10),
            ("cebra-t", f"figure2_update_v6/cebra-time-allrats*.json", 10),
            ("pivae-wo", f"figure2_update_v5/pivae-without-labels_*.json", 10),
            ("tsne", f"figure2_update_v6/tsne-allrats*", 10),
            ("umap", f"figure2_update_v6/umap-allrats*", 10),
            # ("pca", f"{root}/figure2_update_v6/pca*", 1),
        ]
    }


# This is the final figure used for the paper (?)
def results_v3():
    return {
        key: _load(key, fname, num_seeds=num_seeds)
        for key, fname, num_seeds in [
            # ("cebra-1-b", f"{root}/figure2_update_v6/cebra-behavior-step1-sweep_*.json", 3),
            (
                "cebra-10-b",
                f"figure2_update_v5/cebra-behavior-allrats*.json",
                10,
            ),
            # ("cebra-1-t", f"{root}/figure2_update_v6/cebra-time-step1-sweep_*.json", 3),
            # ("cebra-1-sweep-t", f"{root}/figure2_update_v6/cebra-time-step1-sweep-model_*.json", 3),
            # ("cebra-1-sweep2-t", f"{root}/figure2_update_v6/cebra-time-step1-sweep-model-v2*.json", 3),
            # ("cebra-1-dims", f"{root}/figure2_update_v6/cebra-time-step1-sweep-dim*.json", 3),
            ("cebra-10-t", f"figure2_update_v5/cebra-time-allrats*.json", 10),
            # ("pivae-1-w", f"{root}/figure2_update_v6/pivae-step1-with-labels_*.json", 10),
            ("pivae-10-w", f"figure2_update_v6/pivae-with-labels_*.json", 10),
            # ("pivae-1-wo", f"{root}/figure2_update_v6/pivae-step1-without-labels_*.json", 10),
            (
                "pivae-10-wo",
                f"figure2_update_v6/pivae-without-labels_*.json",
                10,
            ),
            ("tsne", f"figure2_update_v6/tsne-allrats*", 10),
            ("umap", f"figure2_update_v6/umap-allrats*", 10),
            ("pca", f"figure2_update_v5/pca*", 1),
        ]
    }


def results_v4():
    return {
        key: _load(key, fname, num_seeds=num_seeds)
        for key, fname, num_seeds in [
            (
                "cebra-10o-b",
                f"figure2_update_v5/cebra-behavior-allrats*.json",
                10,
            ),
            (
                "cebra-10-b",
                f"figure2_update_v7/cebra-behavior-allrats*.json",
                10,
            ),
            ("cebra-10o-t", f"figure2_update_v5/cebra-time-allrats*.json", 10),
            ("cebra-10-t", f"figure2_update_v7/cebra-time-allrats*.json", 10),
            (
                "pivae-10-wo",
                f"figure2_update_v7/pivae-without-labels_*.json",
                10,
            ),
            ("tsne", f"figure2_update_v7/tsne-allrats*", 10),
            ("umap", f"figure2_update_v7/umap-allrats*", 10),
            ("pca", f"figure2_update_v7/pca*", 1),
        ]
    }

if __name__ == '__main__':
    main(results_v1)
    main(results_v2)
    main(results_v3)
    main(results_v4)