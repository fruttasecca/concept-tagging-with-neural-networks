#!/usr/bin/python3
import getopt
import sys

import pandas as pd
import numpy as np
import gensim
from sklearn.decomposition import PCA  # for reducing dimensionality of result


def parse_args(args):
    try:
        opts, args = getopt.getopt(args, "", ["embedding=", "google="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)
        sys.exit(2)

    opts = dict(opts)
    if "--help" in opts:
        print("Parameters:")
        print("--embedding= Which word embeddings to use as a starting point")
        print("Optional:")
        print("--google= To provide the google embeddings bin (and ignore the pickle")
        print("--help= to print this message again")
        exit(0)

    embedding = opts.get("--embedding", None)
    assert embedding is not None or "--google" in opts, "Provide a word embeddings file (pickle) or a google bin."

    google = opts.get("--google", None)

    res = dict()
    res["embedding"] = embedding
    res["google"] = google

    print("-------------")
    print("Running with the following params:")
    for param in sorted(res):
        print("%s = %s" % (param, res[param]))
    print("-------------")
    return res


def save_to_df(chars_list, weights, path):
    res = pd.DataFrame()
    chars = []
    vectors = []

    for idx, c in enumerate(chars_list):
        chars.append(c)
        v = np.array(weights[idx, :])
        vectors.append(v)
    res["token"] = chars
    res["vector"] = vectors

    res.to_pickle(path)


if __name__ == "__main__":
    params = parse_args(sys.argv[1:])
    if not params["google"]:
        w2v_df = pd.read_pickle(params["embedding"])
        starting_dim = len(w2v_df.head(1)["vector"][0])
    else:
        starting_dim = 300

    # map each character to a pair (sum of word vectors the character is in, number of word vectors the character is in)
    c2v = {}
    if params["google"] is not None:
        path = params["google"]
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        keys = embedding_model.vocab
        weights = embedding_model.syn0

        for token in keys:
            v = weights[keys[token].index]
            for c in token:
                if ord(c) < 128:  # only ascii
                    if c in c2v:
                        # update (accumulated vector, count) pair
                        c2v[c] = (c2v[c][0] + v, c2v[c][1] + 1)
                    else:
                        c2v[c] = (v, 1)

    else:
        for index, data_point in w2v_df.iterrows():
            token = data_point["token"]
            v = np.array(data_point["vector"])

            # accumulate vector for every character
            for c in token:
                if ord(c) < 128:  # only ascii
                    if c in c2v:
                        # update (accumulated vector, count) pair
                        c2v[c] = (c2v[c][0] + v, c2v[c][1] + 1)
                    else:
                        c2v[c] = (v, 1)
    # average
    c2v_weights = np.zeros(shape=(len(c2v), starting_dim))
    for idx, (c, (v, count)) in enumerate(sorted(c2v.items())):
        avg_v = np.round(v / count, 5)
        c2v_weights[idx, :] = np.array(avg_v)

    dims = [5, 10, 20, 50, 100, 200, 300]
    for dim in dims:
        if dim <= starting_dim:
            pca = PCA(dim)
            res = pca.fit_transform(c2v_weights)
            save_to_df(sorted(c2v.keys()), res, "../data/c2v_%i.pickle" % dim)
