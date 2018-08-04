#!/usr/bin/python3
import os
import random
import itertools

import pandas as pd
import numpy as np
import time
import getopt
import sys
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import data_manager
from data_manager import PytorchDataset, w2v_matrix_vocab_generator
from models import lstm, gru, rnn, lstm2ch, encoder, attention, conv, fcinit, lstmcrf


def worker_init(*args):
    """
    Init functions for data loader workers.
    :param args:
    :return:
    """
    random.seed(1337)
    np.random.seed(1337)


def predict(model, data_to_predict):
    """
    Use the model to predict on data.
    :param model: The nn module (or equivalent, implementing zero_grad() and being callable).
    :param data_to_predict: PytorchDataset containing data.
    :return:
    """
    y_predicted = []

    dataloader = DataLoader(data_to_predict, 1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x, worker_init_fn=worker_init)
    for batch in dataloader:
        current = []

        # predict and check error
        predicted, _ = model(batch)

        # needed because other models return a score for each possible tag class
        if not isinstance(model, lstmcrf.LstmCrf):
            predicted = torch.argmax(predicted, dim=1)

        for i in predicted:
            current.append(i.item())
        y_predicted.append(current)
    return y_predicted


def write_predictions(tokens, labels, predictions, path, is_indexes, class_dict):
    """
    Write predictions to file, 1 word per line format.
    :param tokens: Word tokens of sentences, a list of lists (a list of sentences).
    :param labels: Concepts/labels of sentences, a list of lists, if is_indexes is True these must be
    concept indices instead of strings, to be mapped back to string with the class_dict.
    :param predictions: Indexes representing classes, a list of lists, mapped back to concepts (strings) with class_dict.
    :param path: where to save the predictions.
    """
    index_to_class = {v: k for k, v in class_dict.items()}
    with open(path, "w") as file:
        for tokens_seq, labels_seq, predictions_seq in zip(tokens, labels, predictions):
            for word, concept, predicted_concept in zip(tokens_seq, labels_seq, predictions_seq):
                conc = index_to_class[concept] if is_indexes else concept
                file.write("%s %s %s\n" % (word, conc, index_to_class[predicted_concept]))
            file.write("\n")


def evaluate_model(dev_data, model, class_dict, batch_size):
    """
    Test a model on data and print the error, precision, recall and f1 score.

    :param dev_data: Data on which to train.
    :param model: The nn module (or equivalent, implementing zero_grad() and being callable).
    :param class_dict: Dict mapping indices to concepts.
    :param batch_size: Size of the training batch.
    """
    error = []
    y_predicted = []
    y_true = []

    dataloader = DataLoader(dev_data, batch_size, shuffle=False, num_workers=1, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x, worker_init_fn=worker_init)

    for batch in dataloader:

        # predict and check error
        predicted, labels = model(batch)

        # needed because other models return a score for each possible tag class
        if not isinstance(model, lstmcrf.LstmCrf):
            loss = torch.nn.functional.nll_loss(predicted, labels, ignore_index=-1)
            # update current epoch dev_data
            error.append(loss.item())
            predicted = torch.argmax(predicted, dim=1)

        # add labels and predictions to list
        tmp_pred = []
        tmp_true = []
        for index, label in zip(predicted, labels):
            ival = index.item()
            labelval = label.item()
            if labelval != -1:
                tmp_pred.append(ival)
                tmp_true.append(labelval)
            else:
                y_predicted.append(tmp_pred)
                y_true.append(tmp_true)
                tmp_pred, tmp_true = [], []

    if not isinstance(model, lstmcrf.LstmCrf):
        print("Dev   error: %f" % np.mean(error))

    # evaluate by calling the evaluation script then clean up
    print("Dev stats:")
    write_predictions(y_true, y_true, y_predicted, "../output/dev_pred.txt", True, class_dict)
    os.system("../output/conlleval.pl < ../output/dev_pred.txt | head -n2")
    os.system("rm ../output/dev_pred.txt")


def train_model(train_data, model, class_dict, dev_data, batch_size, lr, epochs, decay=0.0):
    """
    Trains a model and prints error, precision, recall and f1 while doing so, if dev data is passed
    the model is going to be evaluated on it every epoch.
    :param train_data: Data on which to train.
    :param model: The nn module (or equivalent, implementing zero_grad() and being callable).
    :param class_dict: Dict mapping indices to concepts.
    :param dev_data: Dev data on which to evaluate, if this is passed the function will also print f1 and error for both
    train and dev data.
    :param batch_size: Size of the training batch.
    :param lr: Learning rate.
    :param epochs: Epochs on the data set.
    :param decay: L2 norm decay to be used, default is 0.
    """
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True,
                                 weight_decay=decay)
    # to adjust the lr
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

    starting_time = time.time()

    dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=1, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x, worker_init_fn=worker_init)

    for epoch in range(epochs):
        # setup current epoch train_data
        error = []
        y_predicted = []
        y_true = []

        start = time.time()

        # train
        model.zero_grad()
        for batch in dataloader:

            # predict and check error
            if isinstance(model, lstmcrf.LstmCrf):
                loss = model.neg_log_likelihood(batch)
            else:
                predicted, labels = model(batch)
                loss = torch.nn.functional.nll_loss(predicted, labels, ignore_index=-1)
                indices = torch.argmax(predicted, dim=1)

                # add labels and predictions to list
                tmp_pred = []
                tmp_true = []
                for index, label in zip(indices, labels):
                    ival = index.item()
                    labelval = label.item()
                    if labelval != -1:
                        tmp_pred.append(ival)
                        tmp_true.append(labelval)
                    else:
                        y_predicted.append(tmp_pred)
                        y_true.append(tmp_true)
                        tmp_pred, tmp_true = [], []

            loss.backward()
            optimizer.step()
            model.zero_grad()
            # update current epoch train_data
            error.append(loss.item())

        scheduler.step(np.mean(error))

        print("----- Training epoch stats for epoch %i -----" % epoch)
        print("Seconds for epoch: % f" % (time.time() - start))

        print("Train error: %f" % np.mean(error))
        if not isinstance(model, lstmcrf.LstmCrf):
            print("Train stats:")
            # evaluate by calling the evaluation script then clean up
            write_predictions(y_true, y_true, y_predicted, "../output/train_pred.txt", True,
                              class_dict)
            os.system("../output/conlleval.pl < ../output/train_pred.txt | head -n2")
            os.system("rm ../output/train_pred.txt")

        # if we passed dev train_data to it evaluate on it and report, else keep training
        if dev_data is not None:
            model.eval()
            evaluate_model(dev_data, model, class_dict, batch_size)
            model.train()

    print("total time")
    print(time.time() - starting_time)


def explain_usage(models):
    print("Usage:")
    print("./run_model --train=<train pickle> --test=<test pickle> --w2v=<w2v embeddings pickle> --model=<model> "
          "<rest of params>")
    print("Where model is among: %s" % models)
    print("Arguments that can also be used:")
    print("--c2v=<path to c2v embeddings pickle, if the selected model supports character level information, "
          "like lstm or lstmcrf, the embeddings are going to be used in addition with information from w2v "
          "embeddings, see the paper for more info.")
    print("--write_results=<path> to save the prediction on test data to the specified position, in 1 word per line "
          "format")
    print("--save_model=<path> to save the trained model to the specified position")
    print("--dev to check F1, precision, recall, error on the test set after every epoch")
    print("--help to repeat this message")
    print("Arguments that can also be used (hyperparameters):")
    print("--batch=<batch size>, defaults to 20")
    print("--bidirectional to make it so that recurrent layers will be bidirectional, default is false")
    print("--unfreeze to make it so that w2v embeddings are trained/modified during training, default is false")
    print("--decay=<decay>, decay for l2 normalization, default is 0.0")
    print("--drop=<drop>, drop rate for dropout layers, default is 0.0")
    print("--embedding_norm=<embedding_norm>, max norm of the embeddings if they are trained during training, "
          "with --unfreeze, default is 10")
    print("--epochs=<number of epochs>, defaults to 20")
    print("--hidden_size=<hidden_size>, hidden size for the recurrent layer of any model, default is 200")
    print("--lr=<learning rate>, defaults is 0.001")


def parse_args(args):
    """
    :param args: String of arguments (obtained from sys...), see the "explain_usage" function or directly run
    the script with --help for more info.
    :return: Dictionary mapping a parameter to a value,
    """
    try:
        opts, args = getopt.getopt(args, "",
                                   ["train=", "test=", "w2v=", "model=", "c2v=", "write_results=", "save_model=", "dev",
                                    "help", "batch=", "bidirectional", "unfreeze", "decay=", "drop=", "embedding_norm=",
                                    "epochs=", "hidden_size=", "lr="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)
        sys.exit(2)

    # possible values for target classes, possible values for models to use, possible modes
    possible_models = ["lstm", "rnn", "gru", "lstm2ch", "encoder", "attention", "conv", "fcinit", "lstmcrf"]

    opts = dict(opts)
    if "--help" in opts:
        explain_usage(possible_models)
        exit(0)

    # check args
    train = opts.get("--train", "")
    assert os.path.isfile(train), "train pickle is not there"

    test = opts.get("--test", "")
    assert os.path.isfile(test), "test pickle is not there"

    w2v = opts.get("--w2v", "")
    assert os.path.isfile(w2v), "w2v embeddings pickle is not there"

    model = opts.get("--model", "")
    assert model in possible_models, "use a model from the possible models:\n %s" % possible_models

    c2v = opts.get("--c2v", None)
    if c2v is not None:
        assert os.path.isfile(c2v), "c2v embeddings pickle is not there"

    save_model = opts.get("--save_model", None)
    write_results = opts.get("--write_results", None)
    dev = "--dev" in opts

    batch = int(opts.get("--batch", 20))
    assert batch > 0, "batch size should be greater than 0"

    bidirectional = "--bidirectional" in opts
    unfreeze = "--unfreeze" in opts

    decay = float(opts.get("--decay", 0.00))
    assert decay >= 0, "decay should be greater or equal to 0"

    drop = float(opts.get("--drop", 0.00))
    assert drop >= 0, "dropout rate should be greater or equal to 0"

    embedding_norm = float(opts.get("--embedding_norm", 10.00))
    assert embedding_norm >= 0, "embedding_norm should be greater or equal to 0"

    epochs = int(opts.get("--epochs", 20))
    assert epochs > 0, "epochs size should be greater than 0"

    hidden_size = int(opts.get("--hidden_size", 200))
    assert hidden_size > 0, "hidden size should be greater than 0"
    assert (bidirectional and hidden_size % 2 == 0) or (not bidirectional), "hidden size must be even if " \
                                                                            "--bidirectional is used "

    lr = float(opts.get("--lr", 0.001))
    assert lr > 0, "learning rate should be greater than 0"

    res = dict()
    res["train"] = train
    res["test"] = test
    res["w2v"] = w2v
    res["model"] = model
    res["c2v"] = c2v
    res["save_model"] = save_model
    res["write_results"] = write_results
    res["dev"] = dev
    res["batch"] = batch
    res["bidirectional"] = bidirectional
    res["unfreeze"] = unfreeze
    res["decay"] = decay
    res["drop"] = drop
    res["embedding_norm"] = embedding_norm
    res["epochs"] = epochs
    res["hidden_size"] = hidden_size
    res["lr"] = lr

    print("-------------")
    print("Running with the following params:")
    for param in sorted(res):
        print("%s = %s" % (param, res[param]))
    print("-------------")
    return res


def generate_class_dict(train_df, test_df):
    """
    Given the train and test dataframe, containing "concepts" columns, where every entry is a list of strings representing
    the concepts or classes we are trying to predict, return a dictionary mapping a concept to a index.
    :param train_df: Train dataframe, must contain the "concepts" column.
    :param test_df: Test dataframe, must contain the "concepts" column.
    :return:
    """
    class_dict = dict()
    # make a set of concepts by merging the sets obtained by concepts from train and test dataframes
    concepts = set(itertools.chain(*train_df["concepts"].values)) | set(itertools.chain(*test_df["concepts"].values))
    # add to dict and return
    for concept in sorted(concepts):
        class_dict[concept] = len(class_dict)
    return class_dict


def generate_model_and_transformers(params, class_dict):
    """
    Pick and construct the model and the init and drop transformers given the params, the init transformer
    makes it so that the data in the PytorchDataset is in the tensors of shape and sizes needed, the drop transformer
    randomly drops tokens at run time when a sample is returned from the dataset, to simulate unknown words.
    Also deals with selecting the right device and putting the model on that device, GPU is preferred if available.
    :return: model, data transformer at dataset initialization, data transformer at run time
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w2v_vocab, w2v_weights = w2v_matrix_vocab_generator(params["w2v"])
    c2v_vocab = None
    c2v_weights = None

    if params["c2v"] is not None:
        c2v_vocab, c2v_weights = w2v_matrix_vocab_generator(params["c2v"])

    init_data_transform = data_manager.InitTransform(w2v_vocab, class_dict, c2v_vocab)
    drop_data_transform = data_manager.DropTransform(0.001, w2v_vocab["<UNK>"], w2v_vocab["<padding>"])

    # needed for some models, given their architecture, i.e. CONV
    padded_sentence_length = 50
    # needed by models when using c2v embeddings
    padded_word_length = 30
    if params["model"] == "lstm":
        model = lstm.LSTM(device, w2v_weights, params["hidden_size"], len(class_dict),
                          params["drop"],
                          params["bidirectional"], not params["unfreeze"], params["embedding_norm"],
                          c2v_weights, padded_word_length)
    elif params["model"] == "gru":
        model = gru.GRU(device, w2v_weights, params["hidden_size"], len(class_dict),
                        params["drop"],
                        params["bidirectional"], not params["unfreeze"], params["embedding_norm"],
                        c2v_weights, padded_word_length)
    elif params["model"] == "rnn":
        model = rnn.RNN(device, w2v_weights, params["hidden_size"], len(class_dict),
                        params["drop"],
                        params["bidirectional"], not params["unfreeze"], params["embedding_norm"],
                        c2v_weights, padded_word_length)
    elif params["model"] == "lstm2ch":
        model = lstm2ch.LSTM2CH(device, w2v_weights, params["hidden_size"], len(class_dict), params["drop"],
                                params["bidirectional"], params["embedding_norm"])
    elif params["model"] == "encoder":
        tag_embedding_size = 20
        model = encoder.EncoderDecoderRNN(device, w2v_weights, tag_embedding_size, params["hidden_size"],
                                          len(class_dict), params["drop"], params["bidirectional"],
                                          not params["unfreeze"], params["embedding_norm"],
                                          params["embedding_norm"])
    elif params["model"] == "attention":
        tag_embedding_size = 20
        model = attention.Attention(device, w2v_weights, tag_embedding_size, params["hidden_size"],
                                    len(class_dict), params["drop"], params["bidirectional"], not params["unfreeze"],
                                    params["embedding_norm"], params["embedding_norm"],
                                    padded_sentence_length=padded_sentence_length)
    elif params["model"] == "conv":
        model = conv.CONV(device, w2v_weights, params["hidden_size"], len(class_dict), padded_sentence_length,
                          params["drop"], params["bidirectional"], not params["unfreeze"],
                          params["embedding_norm"])
    elif params["model"] == "fcinit":
        model = fcinit.FCINIT(device, w2v_weights, params["hidden_size"], len(class_dict), padded_sentence_length,
                              params["drop"], params["bidirectional"], not params["unfreeze"], params["embedding_norm"])
    elif params["model"] == "lstmcrf":
        model = lstmcrf.LstmCrf(device, w2v_weights, class_dict, params["hidden_size"], params["drop"],
                                params["bidirectional"], not params["unfreeze"], params["embedding_norm"], c2v_weights,
                                padded_word_length)

    model = model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("total trainable parameters %i" % params)
    return model, init_data_transform, drop_data_transform


if __name__ == "__main__":
    random.seed(1337)
    np.random.seed(1337)
    params = parse_args(sys.argv[1:])

    # load data
    print("loading data")
    train_df = pd.read_pickle(params["train"])
    test_df = pd.read_pickle(params["test"])
    class_dict = generate_class_dict(train_df, test_df)

    # build model and data transformers based on arguments
    model, init_data_transform, run_data_transform = generate_model_and_transformers(params, class_dict)

    train_data = PytorchDataset(train_df, init_data_transform, run_data_transform)
    test_data = PytorchDataset(test_df, init_data_transform)  # notice that there is no run_data_transform for test data

    if params["dev"]:
        print("training in dev mode")
        train_model(train_data, model, class_dict, test_data, params["batch"], params["lr"], params["epochs"],
                    params["decay"])
    else:
        print("training")
        train_model(train_data, model, class_dict, None, params["batch"], params["lr"], params["epochs"],
                    params["decay"])

    print("testing")
    model.eval()
    predictions = predict(model, test_data)
    if params["write_results"] is not None:
        write_predictions(test_df["tokens"].values, test_df["concepts"].values, predictions,
                          params["write_results"], False, class_dict)

    if params["save_model"] is not None:
        torch.save(model.state_dict(), params["save_model"])
