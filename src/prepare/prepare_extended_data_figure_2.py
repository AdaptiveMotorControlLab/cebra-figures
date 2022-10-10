"""Format CEBRA logs for plotting extended data figure 2."""
import cebra.datasets

import numpy as np
import scipy.linalg
import pandas as pd

import joblib
import sklearn


class OrthogonalProcrustesAlignment:
    """Aligns two embedding spaces with an orthogonal map."""

    def __init__(self, top_k=5, subsample=500):
        self.subsample = subsample
        self.top_k = top_k

    def _distance(self, label_i, label_j):
        norm_i = (label_i**2).sum(1)
        norm_j = (label_j**2).sum(1)
        diff = np.einsum("nd,md->nm", label_i, label_j)
        diff = norm_i[:, None] + norm_j[None, :] - 2 * diff
        return diff

    def fit(self, data_i, data_j, label_i, label_j):
        """Aligns dataset j to the reference dataset i."""

        distance = self._distance(label_j, label_i)
        target_idx = np.argsort(distance, axis=1)[:, : self.top_k]

        X = data_j[:, None].repeat(5, axis=1).reshape(-1, 3)
        Y = data_i[target_idx].reshape(-1, 3)
        if self.subsample is not None:
            idc = np.random.choice(len(X), 500)
            X = X[idc]
            Y = Y[idc]

        self._transform, _ = scipy.linalg.orthogonal_procrustes(X, Y)

        return self

    def transform(self, data):
        if self.transform is None:
            raise ValueError("Call fit() first.")
        return data @ self._transform

    def fit_transform(self, data_i, data_j, label_i, label_j):
        self.fit(data_i, data_j, label_i, label_j)
        return self.transform(data_j)


def compute_and_align_embedding():
    """Pre-processing code for results to prepare for plotting.

    NOTE(stes): For demo purposes; requires access to raw data (if you need access
    to it for reproducing something specific, please open a github issue)
    """

    embedding_fnames = [
        "/home/stes/logs/figure2_update_epfl_final/cebra-time-lowertemp/900580/001/emission.jl",
        "/home/stes/logs/figure2_update_epfl_final/cebra-time-lowertemp/900580/004/emission.jl",
        "/home/stes/logs/figure2_update_epfl_final/cebra-time-lowertemp/900580/007/emission.jl",
        "/home/stes/logs/figure2_update_epfl_final/cebra-time-allrats/900438/001/emission.jl",
    ]

    embeddings = []

    for fname in embedding_fnames:
        checkpoint = joblib.load(fname)
        args = checkpoint[0]["args"]
        embedding = checkpoint[0]["emission"]
        ica = (
            sklearn.decomposition.FastICA(2, whiten=True)
            .fit(embedding)
            .transform(embedding)
        )
        labels = cebra.datasets.init(
            args["dataset"], root="/home/stes/projects/neural_cl/cebra_public/data"
        ).index
        labels = labels.numpy()
        if len(embeddings) >= 1:
            _, ref_embedding, _, ref_ica, _ = embeddings[0]
            transform, _ = scipy.linalg.orthogonal_procrustes(embedding, ref_embedding)
            aligned_embedding = embedding @ transform
            transform, _ = scipy.linalg.orthogonal_procrustes(ica, ref_ica)
            ica = ica @ transform
        else:
            aligned_embedding = embedding

        embeddings.append((args, embedding, aligned_embedding, ica, labels))

    return embeddings


def save_embeddings(embeddings):
    embeddings_df = pd.DataFrame(
        [
            dict(
                rat_id=rat_id,
                args=args,
                embedding=embedding,
                data=data,
                ica=ica,
                labels=labels,
            )
            for rat_id, (args, embedding, data, ica, labels) in enumerate(embeddings)
        ]
    )
    embeddings_df.to_hdf("../data/EDFigure2.h5", key="data")


embeddings = compute_and_align_embedding()
save_embeddings(embeddings)
