from annoy import AnnoyIndex


def build_index(embedding_l, metric='euclidean', n_trees=50):
    # TODO: supply docstring
    f = len(embedding_l[0])
    t = AnnoyIndex(f, metric)
    for i, vec in enumerate(embedding_l):
        t.add_item(i, vec)

    t.build(n_trees)
    return t


def assessor(annoy_index, query_vec, k=-1, include_distances=True):
    # TODO: supply docstring
    if k == -1:
        k = annoy_index.get_n_items()

    sims = annoy_index.get_nns_by_vector(query_vec, n=k, include_distances=include_distances)
    return sims
