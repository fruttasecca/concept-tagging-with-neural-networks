#!/usr/bin/python3
import os

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
from models import lstm, gru, rnn, lstm_2ch, encoder, attention, conv, init_hidden, lstmcrf


def predict(model, train_data):
    """
    Use the model to predict on data.
    :param model:
    :param train_data:
    :return:
    """
    y_predicted = []

    dataloader = DataLoader(train_data, 1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x)
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


def write_predictions(tokens, labels, predictions, path, is_indexes, class_vocab):
    """
    Write predictions to file.
    :param tokens:
    :param labels:
    :param predictions:
    :param path:
    :param indexed_labels:
    :return:
    """
    index_to_class = {v: k for k, v in class_vocab.items()}
    with open(path, "w") as file:
        for tokens_seq, labels_seq, predictions_seq in zip(tokens, labels, predictions):
            for word, concept, predicted_concept in zip(tokens_seq, labels_seq, predictions_seq):
                conc = index_to_class[concept] if is_indexes else concept
                file.write("%s %s %s\n" % (word, conc, index_to_class[predicted_concept]))
            file.write("\n")


def evaluate_model(dev_data, model, class_vocab, batch_size):
    """
    Test a model on data and print the error, precision, recall and f1 score.

    :param dev_data: Data on which to train.
    :param model: The nn module (or equivalent, implementing zero_grad() and being callable).
    :param batch_size: Size of the training batch.
    """
    error = []
    y_predicted = []
    y_true = []

    dataloader = DataLoader(dev_data, batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x)

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
    write_predictions(y_true, y_true, y_predicted, "../output/dev_pred.txt", True, class_vocab)
    os.system("../output/%s/conlleval.pl < ../output/dev_pred.txt | head -n2" % params["dataset"])
    os.system("rm ../output/dev_pred.txt")


def train_model(train_data, model, class_vocab, dev_data=None, batch_size=80, lr=0.0001, epochs=20, decay=0.0):
    """
    Trains a model and prints error, precision, recall and f1 while doing so, if dev data is passed
    the model is going to be evaluated on it every epoch.
    :param train_data: Data on which to train.
    :param model: The nn module (or equivalent, implementing zero_grad() and being callable).
    :param dev_data: Dev data on which to evaluate, if this is passed the function will also print f1 and error for both
    train and dev data, moreover a plot of the train and dev error will be saved.
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

    dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x)

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
            write_predictions(y_true, y_true, y_predicted, "../output/%s/train_pred.txt" % params["dataset"], True,
                              class_vocab)
            os.system("../output/%s/conlleval.pl < ../output/%s/train_pred.txt | head -n2" % (
                params["dataset"], params["dataset"]))
            os.system("rm ../output/%s/train_pred.txt" % params["dataset"])

        # if we passed dev train_data to it evaluate on it and report, else keep training
        if dev_data is not None:
            model.eval()
            evaluate_model(dev_data, model, class_vocab, batch_size)
            model.train()

    print("total time")
    print(time.time() - starting_time)


def explain_usage(models):
    print("usage:")
    print("./run_model --model=<model>")
    print("Where model is among: %s" % models)
    print("Optional arguments that can also be used:")
    print("--lr=<learning rate>, defaults is 0.01")
    print("--drop=<drop>, drop rate for dropout layers, default is 0.0")
    print("--decay=<decay>, decay for l2 normalization, default is 0.0")
    print("--embedding_norm=<embedding_norm>, max norm of the embeddings if they are trained during training, "
          "either because of the --unfreeze parameter of because of the specific model, default is 10")
    print("--batch=<batch size>, defaults to 80")
    print("--epochs=<number of epochs>, defaults to 30")
    print("--hidden_size=<hidden_size>, hidden size for the recurrent layer of any model")
    print("--dev to train  and test against dev set (0.15% of data) at every epoch")
    print("--save_model=<path> to save the trained model to the specified position")
    print("--write_results=<path> to save the prediction on test data to the specified position, in 1 word per line "
          "format")
    print("--bidirectional to make it so that recurrent layers will be bidirectional, default is false")
    print("--unfreeze to make it so that embedding are trained during training, even if import from pre-trained "
          "embeddings, default is false")
    print(
        "--char_embedding=<path_to_char_embedding By using the recurrent model and providing this the recurrent model will "
        "use char level embeddings in conjunction with the word embeddings, if the model 'reccurent' is not in use this will be "
        "ignored")


def parse_args(args):
    """
    :param args: String of arguments (obtained from sys...)
    :return: Dictionary mapping a parameter to a value,
    lr = learning rate
    drop = drop
    decay = decay
    embedding_norm = embedding_norm
    batch = batch
    epochs = epochs
    hidden_size = hidden_size
    model = which model to use
    save_model = where to save the model if it is to be saved
    write_results = where to save the predictions on test data, has no effect in dev mode
    dev = if train in dev mode
    bidirectional = if the recurrent model should be bidirectional
    unfreeze = if embedding should be unfrozen and trained
    char_embedding = which char embeddings to use (providing this will make it so that the recurrent model will use char embeddings)
    """
    try:
        opts, args = getopt.getopt(args, "",
                                   ["model=", "batch=", "epochs=", "lr=", "dev", "save_model=", "class=", "help",
                                    "sequence=",
                                    "vector", "write_results=", "bidirectional", "hidden_size=", "unfreeze",
                                    "drop=", "decay=", "embedding_norm=", "char_embedding=", "dataset="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)
        sys.exit(2)

    # possible values for target classes, possible values for models to use, possible modes
    possible_models = ["lstm", "rnn", "gru", "lstm2ch", "encoder", "attention", "conv", "init_hidden", "lstmcrf"]
    possible_datasets = ["movies", "atis"]

    opts = dict(opts)
    if "--help" in opts:
        explain_usage(possible_models)
        exit(0)

    lr = float(opts.get("--lr", 0.01))
    assert lr > 0, "learning rate should be greater than 0"

    drop = float(opts.get("--drop", 0.00))
    assert drop >= 0, "dropout rate should be greater or equal to 0"

    decay = float(opts.get("--decay", 0.00))
    assert decay >= 0, "decay should be greater or equal to 0"

    embedding_norm = float(opts.get("--embedding_norm", 10.00))
    assert embedding_norm >= 0, "embedding_norm should be greater or equal to 0"

    batch = int(opts.get("--batch", 80))
    assert batch > 0, "batch size should be greater than 0"

    epochs = int(opts.get("--epochs", 30))
    assert epochs > 0, "epochs size should be greater than 0"

    hidden_size = int(opts.get("--hidden_size", 200))
    assert hidden_size > 0, "hidden size should be greater than 0"

    model = opts.get("--model", "")
    assert model in possible_models, "use a model from the possible models:\n %s" % possible_models

    dataset = opts.get("--dataset", "")
    assert dataset in possible_datasets, "use a dataset from the possible datasets:\n %s" % possible_datasets

    res = dict()
    res["lr"] = lr
    res["drop"] = drop
    res["decay"] = decay
    res["embedding_norm"] = embedding_norm
    res["batch"] = batch
    res["epochs"] = epochs
    res["hidden_size"] = hidden_size
    res["model"] = model
    res["dataset"] = dataset
    res["save_model"] = opts.get("--save_model", None)
    res["write_results"] = opts.get("--write_results", None)
    res["dev"] = "--dev" in opts
    res["bidirectional"] = "--bidirectional" in opts
    res["unfreeze"] = "--unfreeze" in opts
    if dataset == "movies":
        res["embedding"] = "../data/movies/w2v_trimmed.pickle"
    elif dataset == "atis":
        res["embedding"] = "../data/atis/w2v_trimmed.pickle"

    res["char_embedding"] = opts.get("--char_embedding", None)

    print("-------------")
    print("Running with the following params:")
    for param in sorted(res):
        print("%s = %s" % (param, res[param]))
    print("-------------")
    return res


def pick_model_transform(params):
    """
    Pick and construct the model and the init and drop transforms given the params.
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w2v_vocab, w2v_weights = w2v_matrix_vocab_generator(params["embedding"])
    c2v_vocab = None
    c2v_weights = None
    if params["dataset"] == "movies":
        class_vocab = data_manager.class_vocab_movies
    elif params["dataset"] == "atis":
        class_vocab = data_manager.class_vocab_atis

    if params["char_embedding"] is not None:
        c2v_vocab, c2v_weights = w2v_matrix_vocab_generator(params["char_embedding"])

    init_data_transform = data_manager.InitTransform(device, w2v_vocab, class_vocab, c2v_vocab)
    drop_data_transform = data_manager.DropTransform(0.001, w2v_vocab["<UNK>"], w2v_vocab["<padding>"])

    if params["model"] == "lstm":
        model = lstm.LSTM(device, w2v_weights, params["hidden_size"], len(class_vocab),
                          params["drop"],
                          params["bidirectional"], not params["unfreeze"], params["embedding_norm"],
                          c2v_weights, 30)
    elif params["model"] == "gru":
        model = gru.GRU(device, w2v_weights, params["hidden_size"], len(class_vocab),
                        params["drop"],
                        params["bidirectional"], not params["unfreeze"], params["embedding_norm"],
                        c2v_weights, 30)
    elif params["model"] == "rnn":
        model = rnn.RNN(device, w2v_weights, params["hidden_size"], len(class_vocab),
                        params["drop"],
                        params["bidirectional"], not params["unfreeze"], params["embedding_norm"],
                        c2v_weights, 30)
    elif params["model"] == "lstm2ch":
        model = lstm_2ch.LSTMN2CH(device, w2v_weights, params["hidden_size"], len(class_vocab), params["drop"],
                                  params["bidirectional"], params["embedding_norm"])
    elif params["model"] == "encoder":
        tag_embedding_size = 20
        model = encoder.EncoderDecoderRNN(w2v_weights, tag_embedding_size, params["hidden_size"],
                                          len(class_vocab), params["drop"], params["bidirectional"],
                                          not params["unfreeze"], params["embedding_norm"],
                                          params["embedding_norm"])
    elif params["model"] == "attention":
        tag_embedding_size = 20
        padded_sentence_length = 50
        model = attention.Attention(w2v_weights, "gru", tag_embedding_size, params["hidden_size"],
                                    len(class_vocab),
                                    params["drop"], params["bidirectional"], not params["unfreeze"],
                                    params["embedding_norm"], params["embedding_norm"],
                                    padded_sentence_length=padded_sentence_length)
    elif params["model"] == "conv":
        padded_sentence_length = 50
        model = conv.CNN(w2v_weights, params["hidden_size"], len(class_vocab), padded_sentence_length,
                         params["drop"], params["bidirectional"], not params["unfreeze"],
                         params["embedding_norm"])
    elif params["model"] == "init_hidden":
        padded_sentence_length = 50
        model = init_hidden.INIT(w2v_weights, params["hidden_size"], len(class_vocab),
                                 padded_sentence_length,
                                 params["drop"], params["bidirectional"], not params["unfreeze"],
                                 params["embedding_norm"])
    elif params["model"] == "lstmcrf":
        model = lstmcrf.LstmCrf(w2v_weights, class_vocab, params["hidden_size"], params["drop"],
                                params["bidirectional"], not params["unfreeze"], params["embedding_norm"], c2v_weights,
                                30)

    model = model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("total trainable parameters %i" % params)
    return model, class_vocab, init_data_transform, drop_data_transform


if __name__ == "__main__":
    params = parse_args(sys.argv[1:])

    # used to transform data once imported
    init_data_transform = None
    # used to transform data on the fly (once get item is called)
    getitem_data_transform = None
    # build model based on argument
    model, class_vocab, init_data_transform, run_data_transform = pick_model_transform(params)

    # run in dev mode
    if params["dev"]:
        if params["dataset"] == "movies":
            train_file = ["../data/movies/train.pickle"]
            dev_file = ["../data/movies/dev.pickle"]

        elif params["dataset"] == "atis":
            train_file = ["../data/atis/train.pickle"]
            dev_file = ["../data/atis/dev.pickle"]

        train_data = PytorchDataset(train_file, init_data_transform, run_data_transform)
        dev_data = PytorchDataset(dev_file, init_data_transform)

        print("training")
        train_model(train_data, model, class_vocab, dev_data, params["batch"], params["lr"], params["epochs"],
                    params["decay"])

        model.eval()
        predictions = predict(model, dev_data)
        dev_pickle = pd.read_pickle(dev_file[0])
        if params["write_results"] is not None:
            write_predictions(dev_pickle["tokens"].values, dev_pickle["concepts"].values, predictions,
                              params["write_results"], False, class_vocab)

        if params["save_model"] is not None:
            torch.save(model.state_dict(), params["save_model"])
    # run in test mode
    else:
        if params["dataset"] == "movies":
            train_files = ["../data/movies/random_bins/train_dev/train_dev_bin6.pickle"]
            test_file = ["../data/movies/test.pickle"]
        elif params["dataset"] == "atis":
            train_files = ["../data/atis/train.pickle", "../data/atis/dev.pickle"]
            test_file = ["../data/atis/test.pickle"]

        train_data = PytorchDataset(train_files, init_data_transform, run_data_transform)
        print("training")
        train_model(train_data, model, class_vocab, None, params["batch"], params["lr"], params["epochs"],
                    params["decay"])
        del train_data

        print("testing")
        test_data = PytorchDataset(test_file, init_data_transform)
        test_pickle = pd.read_pickle(test_file[0])
        model.eval()
        predictions = predict(model, test_data)
        if params["write_results"] is not None:
            write_predictions(test_pickle["tokens"].values, test_pickle["concepts"].values, predictions,
                              params["write_results"], False, class_vocab)
        if params["save_model"] is not None:
            torch.save(model.state_dict(), params["save_model"])
