#!/usr/bin/python3
import getopt
import sys
import os

import pandas as pd
import gensim

from data_manager import Data


def trim_w2v(df, w2v):
    """
    Given a df with tokens, lemmas, pos and concepts columns containing list of strings and a gensim w2v
    model it returns a df mapping a word to its w2v vector if it appears both in our data and in the w2v model.
    Function meant to save time and memory by not loading the whole w2v model every time we run the project.

    :param df: Df containing the columns tokens, lemmas, pos, concepts.
    :param w2v: Gensim w2v KeyedVector model.
    :return: A df mapping symbols ("token" column) to their w2v vector ("vector" column)
    """
    vocab = set()
    for index, data_point in df.iterrows():
        vocab |= set(data_point["tokens"])
        vocab |= set(" ".join(data_point["tokens"]).title().split(" "))
        vocab |= set(data_point["concepts"])
        vocab |= set(" ".join(data_point["concepts"]).title().split(" "))

    # intersection with w2v w2v_vocab
    vocab &= set(w2v.vocab)

    # make a df mapping tokens to vectors
    res = pd.DataFrame()
    tokens = []
    vectors = []

    weights = w2v.syn0
    for token in vocab:
        tokens.append(token)
        vectors.append(weights[w2v.vocab[token].index])
    if "number" not in tokens:
        tokens.append("number")
        vectors.append(weights[w2v.vocab["number"].index])
    if "@card@" not in tokens:
        tokens.append("@card@")
        vectors.append(weights[w2v.vocab["number"].index])
    if "rating" not in tokens:
        tokens.append("rating")
        vectors.append(weights[w2v.vocab["rating"].index])
    res["token"] = tokens
    res["vector"] = vectors
    return res


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "", ["bin=", "help", "dataset="])
    opts = dict(opts)
    dataset = opts.get("--dataset", "")
    if "--help" in opts:
        print("./trim_w2v.py --bin=<google w2v bin> --dataset=dataset")
        exit()
    if "--bin" not in opts:
        print("Please provide the google embeddings bin file.")
        exit()

    train_data = Data("../data/%s/train.data" % dataset).to_dataframe()
    test_data = Data("../data/%s/test.data" % dataset).to_dataframe()
    if os.path.exists("../data/%s/dev.data" % dataset):
        dev_data = Data("../data/%s/dev.data" % dataset).to_dataframe()
        total_data = pd.concat([train_data, dev_data, test_data])
    else:
        total_data = pd.concat([train_data, test_data])

    print("loading w2v w2v_weights")
    path = opts.get("--bin", "")
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    print("trimming")
    w2v = trim_w2v(total_data, embedding_model)

    print("writing to disk")
    w2v.to_pickle("../data/%s/w2v_trimmed.pickle" % dataset)
