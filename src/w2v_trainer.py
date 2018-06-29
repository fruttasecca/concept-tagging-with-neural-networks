#!/usr/bin/python3
import torch
import numpy as np
import time
import pandas as pd
import torch.functional as F
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import getopt, sys

import data_manager


class W2V_dataset(Dataset):
    """Dataset to wrap the pairs and return them to the data loader in a more handy way."""

    def __init__(self, data, vocab_size):
        self.data = data
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputidx, outputidx = self.data[idx]
        # input for the nn
        input_v = torch.LongTensor([inputidx])
        # expected result
        expected = torch.LongTensor([outputidx])
        return {"input": input_v, 'expected_idx': expected}


class W2V(nn.Module):
    """
    NN to compute w2v embeddings.
    TODO: optimize so that only the selected vector in the W1 matrix is brought to the gpu/has gradient
    computed on it; also look for hierarchical softmax.
    """

    def __init__(self, vocab_dimension, embedding_dim):
        super(W2V, self).__init__()

        self.W1 = nn.Embedding(vocab_dimension, embedding_dim, max_norm=10)
        self.W2 = nn.Linear(embedding_dim, vocab_dimension)

    def forward(self, batch):
        embedding = self.W1(batch)
        context = self.W2(embedding)
        res = F.log_softmax(context, dim=2)
        return res.view(len(batch), -1)


def skipgram_pairs(df, window_size=20, sequence="tokens"):
    """
    Generates the index pairs to be used as data for training the w2v embeddings, first word of the pair (the index)
    will be the input, the second the output.

    :param df: Dataframe containing a "sequence" column, where each value is a list of strings.
    :param window_size: Size of the skip-gram window.
    :param sequence: Sequence of data on which to do word embeddings (tokens or lemmas or pos).
    :return: (id pairs, word2id dictionary)
    """
    # map unique token types to an idx
    word2id = dict()
    for id, row in df.iterrows():
        for token in row[sequence]:
            if token.isdigit() or token == "@card@":
                token = "number"
            elif token in data_manager.corrected:
                token = data_manager.corrected[token]
            if token not in word2id:
                word2id[token] = len(word2id)

    # generate pairs by "skipgramming" around the currently center word
    id_pairs = []
    window = [w for w in range(-window_size, window_size + 1) if w != 0]
    for _, row in final_df.iterrows():
        # get list of ids forming a sequence
        ids = []
        for token in row[sequence]:
            if token.isdigit() or token == "@card@":
                token = "number"
            elif token in data_manager.corrected:
                token = data_manager.corrected[token]
            ids.append(word2id[token])

        # use each word as the center, and skip-gram around it
        for center_pos, current_id in enumerate(ids):
            # get id at each position in the window and add as a pair
            for skip in window:
                skipped_pos = center_pos + skip
                # check that we are still in the boundaries of the sentence
                if skipped_pos < 0 or skipped_pos >= len(ids):
                    continue
                skipped_id = ids[skipped_pos]
                id_pairs.append((current_id, skipped_id))

    id_pairs = np.array(id_pairs)
    return id_pairs, word2id


def save_to_df(w2id, weights, path):
    res = pd.DataFrame()
    tokens = []
    vectors = []

    total_norm = 0
    for token in w2id:
        tokens.append(token)
        v = weights[w2id[token], :]
        v = np.array(v)
        total_norm += np.linalg.norm(v)
        vectors.append(v)
    res["token"] = tokens
    res["vector"] = vectors

    print("average length of vector")
    print(total_norm / len(w2id))

    res.to_pickle(path)


def parse_args(args):
    try:
        opts, args = getopt.getopt(args, "", ["embedding=", "batch=", "epochs=", "lr=", "help", "window=", "sequence="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)
        sys.exit(2)

    opts = dict(opts)
    if "--help" in opts:
        print("Parameters, all optional:")
        print("--embedding= size of the embedding, an integer")
        print("--batch= batch size")
        print("--epochs= number of epochs")
        print("--window= size of the skip-gram window")
        print("--lr= learning rate")
        print("--help= to print this message again")
        exit(0)

    possible_sequences = ["tokens", "lemmas", "pos", "combined"]
    sequence = opts.get("--sequence", "")
    assert sequence in possible_sequences, "sequence to use should be among the following:\n  %s" % possible_sequences

    embedding_dim = int(opts.get("--embedding", 300))
    assert embedding_dim > 0, "embedding dimension should be greater than 0"

    batch = int(opts.get("--batch", 1000))
    assert batch > 0, "batch size should be greater than 0"

    epochs = int(opts.get("--epochs", 20))
    assert epochs > 0, "epochs size should be greater than 0"

    window = int(opts.get("--window", 20))
    assert window > 0, "skip-gram window size should be greater than 0"

    lr = float(opts.get("--lr", 0.01))
    assert lr > 0, "learning rate should be greater than 0"

    res = dict()
    res["embedding_dim"] = embedding_dim
    res["batch"] = batch
    res["epochs"] = epochs
    res["window"] = window
    res["lr"] = lr
    res["sequence"] = sequence

    print("-------------")
    print("Running with the following params:")
    for param in sorted(res):
        print("%s = %s" % (param, res[param]))
    print("-------------")
    return res


if __name__ == "__main__":
    params = parse_args(sys.argv[1:])

    # import data and generate pairs and word-id dictionary
    train_df = pd.read_pickle("../data/train.pickle")
    dev_df = pd.read_pickle("../data/dev.pickle")
    final_df = pd.concat([train_df, dev_df])

    window = params["window"]
    sequence = params["sequence"]
    id_pairs, word2id = skipgram_pairs(final_df, window, sequence)
    train_data = W2V_dataset(id_pairs, len(word2id))

    embedding_dim = params["embedding_dim"]
    model = W2V(len(word2id), embedding_dim=embedding_dim)
    model.cuda()
    num_epochs = params["epochs"]
    batch_size = params["batch"]
    rate = params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, amsgrad=True)
    # for adjusting the lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.00001, last_epoch=-1)

    for epoch in range(num_epochs):
        epoch_loss = 0
        dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
        starting_time = time.time()

        for batchidx, batch in enumerate(dataloader):
            x = batch["input"].cuda()
            y = batch["expected_idx"].cuda().view(-1)

            model.zero_grad()
            context = model.forward(x)
            loss = F.nll_loss(context, y)
            epoch_loss += loss.data.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print("epoch %i done in %f seconds" % (epoch, time.time() - starting_time))
        print("loss at epoch %i = %f" % (epoch, epoch_loss / len(dataloader)))

    # trained_weights = model.fc1.weight.data
    embeddings = model.W1.weight.data
    save_to_df(word2id, embeddings,
               "../data/w2v_learned_embedding%i_window%i_sequence=%s.pickle" % (embedding_dim, window, sequence))
