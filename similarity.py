from annoy import AnnoyIndex
import numpy as np


def build_index(embedding_l, metric='euclidean', n_trees=50):
    """
    Build a forest of Annoy indices.

    :param embedding_l (list): list of embedding  vectors.
    :param metric (string): metric of similarity. Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot".
    :param n_trees (int): builds a forest of n_trees trees.

    :return AnnoyIndex:
    """
    f = len(embedding_l[0])
    t = AnnoyIndex(f, metric)
    for i, vec in enumerate(embedding_l):
        t.add_item(i, vec)

    t.build(n_trees)
    return t


def assessor(annoy_index, query_vec, k=-1, include_similarity=True):
    """
    Get K nearest neighbors by vector.

    :param annoy_index (AnnoyIndex): Forest of indices.
    :param query_vec (list): Vector of embedding.
    :param k (int): Number of nearest neighbors. If -1, gets all items. default -1.
    :param include_distances (bool): If True, return distances as well.

    :return (list or tuple of lists): sorted indices by similarities also return distances if `include_distances=True`.
    """
    if k == -1:
        k = annoy_index.get_n_items()

    indices, distances = annoy_index.get_nns_by_vector(query_vec, n=k, include_distances=True)
    similarities = list(1 / (1 + np.array(distances)))
    res = (indices, similarities) if include_similarity else indices
    return res
