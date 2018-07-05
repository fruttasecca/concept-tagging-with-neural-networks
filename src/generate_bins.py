#!/usr/bin/python3
import pandas as pd


def frame_to_txt(df, path, replace):
    """
    Write a dataframe as txt data (for wfsts..).
    :param df: Dataframe.
    :param path: Where to save.
    :param replace: If 'O' concepts have to be replaced by tokens.
    :return:
    """
    newpath = path
    if replace:
        newpath += "_replaced.txt"
    else:
        newpath += ".txt"

    if replace:
        with open(newpath, "w") as output:
            for i, row in df.iterrows():
                tokens = row["tokens"]
                concepts = row["concepts"]
                for t, c, in zip(tokens, concepts):
                    if t.isdigit() or t.find("DIGIT") != -1 or any(char.isdigit() for char in t):
                        t = "number"
                    if c == "O":
                        c = t
                    output.write("%s %s\n" % (t, c))
                output.write("\n")
    else:
        with open(newpath, "w") as output:
            for i, row in df.iterrows():
                tokens = row["tokens"]
                concepts = row["concepts"]
                for t, c, in zip(tokens, concepts):
                    if t.isdigit() or t.find("DIGIT") != -1 or any(char.isdigit() for char in t):
                        t = "number"
                    output.write("%s %s\n" % (t, c))
                output.write("\n")

    # concepts phrases
    newpath = path
    if replace:
        newpath += "_concepts_replaced.txt"
    else:
        newpath += "_concepts.txt"

    if replace:
        with open(newpath, "w") as output:
            for i, row in df.iterrows():
                tokens = row["tokens"]
                concepts = row["concepts"]
                for t, c, in zip(tokens, concepts):
                    if t.isdigit() or t.find("DIGIT") != -1 or any(char.isdigit() for char in t):
                        t = "number"
                    if c == "O":
                        c = t
                    output.write("%s " % (c))
                output.write("\n")
    else:
        with open(newpath, "w") as output:
            for i, row in df.iterrows():
                tokens = row["tokens"]
                concepts = row["concepts"]
                for t, c, in zip(tokens, concepts):
                    output.write("%s " % (c))
                output.write("\n")


def generate_bins(dataset):
    if dataset == "movies":
        train = pd.read_pickle("../data/movies/train_split.pickle")
        dev = pd.read_pickle("../data/movies/dev.pickle")
        test = pd.read_pickle("../data/movies/test.pickle")

        dev_share = int(len(dev) / 6)
        # get train bins (to tune on dev)
        train_bin6 = train.sample(2670, random_state=1337)
        train_bin5 = train_bin6.sample(2500 - dev_share * 5, random_state=1337)
        train_bin4 = train_bin5.sample(2000 - dev_share * 4, random_state=1337)
        train_bin3 = train_bin4.sample(1500 - dev_share * 3, random_state=1337)
        train_bin2 = train_bin3.sample(1000 - dev_share * 2, random_state=1337)
        train_bin1 = train_bin2.sample(500 - dev_share, random_state=1337)
        # get dev bins (to have train_dev bins for final training), after tuning
        dev_bin6 = dev.sample(668, random_state=1337)
        dev_bin5 = dev_bin6.sample(dev_share * 5, random_state=1337)
        dev_bin4 = dev_bin5.sample(dev_share * 4, random_state=1337)
        dev_bin3 = dev_bin4.sample(dev_share * 3, random_state=1337)
        dev_bin2 = dev_bin3.sample(dev_share * 2, random_state=1337)
        dev_bin1 = dev_bin2.sample(dev_share, random_state=1337)
    elif dataset == "atis":
        train = pd.read_pickle("../data/atis/train.pickle")
        dev = pd.read_pickle("../data/atis/dev.pickle")
        test = pd.read_pickle("../data/atis/test.pickle")

        # get train bins (to tune on dev)
        dev_share = int(len(dev) / 6)
        train_bin6 = train.sample(3983, random_state=1337)
        train_bin5 = train_bin6.sample(4000 - dev_share * 5, random_state=1337)
        train_bin4 = train_bin5.sample(3000 - dev_share * 4, random_state=1337)
        train_bin3 = train_bin4.sample(2000 - dev_share * 3, random_state=1337)
        train_bin2 = train_bin3.sample(1000 - dev_share * 2, random_state=1337)
        train_bin1 = train_bin2.sample(500 - dev_share, random_state=1337)

        # get dev bins (to have train_dev bins for final training), after tuning
        dev_bin6 = dev.sample(995, random_state=1337)
        dev_bin5 = dev_bin6.sample(dev_share * 5, random_state=1337)
        dev_bin4 = dev_bin5.sample(dev_share * 4, random_state=1337)
        dev_bin3 = dev_bin4.sample(dev_share * 3, random_state=1337)
        dev_bin2 = dev_bin3.sample(dev_share * 2, random_state=1337)
        dev_bin1 = dev_bin2.sample(dev_share, random_state=1337)

    print("train bins")
    print(len(train_bin6))
    print(len(train_bin5))
    print(len(train_bin4))
    print(len(train_bin3))
    print(len(train_bin2))
    print(len(train_bin1))
    train_bin6.to_pickle("../data/%s/random_bins/train/train_bin6.pickle" % dataset)
    train_bin5.to_pickle("../data/%s/random_bins/train/train_bin5.pickle" % dataset)
    train_bin4.to_pickle("../data/%s/random_bins/train/train_bin4.pickle" % dataset)
    train_bin3.to_pickle("../data/%s/random_bins/train/train_bin3.pickle" % dataset)
    train_bin2.to_pickle("../data/%s/random_bins/train/train_bin2.pickle" % dataset)
    train_bin1.to_pickle("../data/%s/random_bins/train/train_bin1.pickle" % dataset)

    print("dev bins")
    print(len(dev_bin6))
    print(len(dev_bin5))
    print(len(dev_bin4))
    print(len(dev_bin3))
    print(len(dev_bin2))
    print(len(dev_bin1))
    dev_bin6.to_pickle("../data/%s/random_bins/dev/dev_bin6.pickle" % dataset)
    dev_bin5.to_pickle("../data/%s/random_bins/dev/dev_bin5.pickle" % dataset)
    dev_bin4.to_pickle("../data/%s/random_bins/dev/dev_bin4.pickle" % dataset)
    dev_bin3.to_pickle("../data/%s/random_bins/dev/dev_bin3.pickle" % dataset)
    dev_bin2.to_pickle("../data/%s/random_bins/dev/dev_bin2.pickle" % dataset)
    dev_bin1.to_pickle("../data/%s/random_bins/dev/dev_bin1.pickle" % dataset)

    print("train_dev_bins")
    # form train_dev bins
    train_dev_bin6 = pd.concat([train_bin6, dev_bin6])
    train_dev_bin5 = pd.concat([train_bin5, dev_bin5])
    train_dev_bin4 = pd.concat([train_bin4, dev_bin4])
    train_dev_bin3 = pd.concat([train_bin3, dev_bin3])
    train_dev_bin2 = pd.concat([train_bin2, dev_bin2])
    train_dev_bin1 = pd.concat([train_bin1, dev_bin1])
    print(len(train_dev_bin6))
    print(len(train_dev_bin5))
    print(len(train_dev_bin4))
    print(len(train_dev_bin3))
    print(len(train_dev_bin2))
    print(len(train_dev_bin1))
    train_dev_bin6.to_pickle("../data/%s/random_bins/train_dev/train_dev_bin6.pickle" % dataset)
    train_dev_bin5.to_pickle("../data/%s/random_bins/train_dev/train_dev_bin5.pickle" % dataset)
    train_dev_bin4.to_pickle("../data/%s/random_bins/train_dev/train_dev_bin4.pickle" % dataset)
    train_dev_bin3.to_pickle("../data/%s/random_bins/train_dev/train_dev_bin3.pickle" % dataset)
    train_dev_bin2.to_pickle("../data/%s/random_bins/train_dev/train_dev_bin2.pickle" % dataset)
    train_dev_bin1.to_pickle("../data/%s/random_bins/train_dev/train_dev_bin1.pickle" % dataset)

    # WFST BINS
    print("writing train bins to .txt for wfsts...")
    frame_to_txt(train_bin6, "../data/%s/random_bins/train/wfst/train_bin6" % dataset, False)
    frame_to_txt(train_bin5, "../data/%s/random_bins/train/wfst/train_bin5" % dataset, False)
    frame_to_txt(train_bin4, "../data/%s/random_bins/train/wfst/train_bin4" % dataset, False)
    frame_to_txt(train_bin3, "../data/%s/random_bins/train/wfst/train_bin3" % dataset, False)
    frame_to_txt(train_bin2, "../data/%s/random_bins/train/wfst/train_bin2" % dataset, False)
    frame_to_txt(train_bin1, "../data/%s/random_bins/train/wfst/train_bin1" % dataset, False)
    frame_to_txt(train_bin6, "../data/%s/random_bins/train/wfst/train_bin6" % dataset, True)
    frame_to_txt(train_bin5, "../data/%s/random_bins/train/wfst/train_bin5" % dataset, True)
    frame_to_txt(train_bin4, "../data/%s/random_bins/train/wfst/train_bin4" % dataset, True)
    frame_to_txt(train_bin3, "../data/%s/random_bins/train/wfst/train_bin3" % dataset, True)
    frame_to_txt(train_bin2, "../data/%s/random_bins/train/wfst/train_bin2" % dataset, True)
    frame_to_txt(train_bin1, "../data/%s/random_bins/train/wfst/train_bin1" % dataset, True)
    print("writing dev bins to .txt for wfsts...")
    frame_to_txt(dev_bin6, "../data/%s/random_bins/dev/wfst/dev_bin6" % dataset, False)
    frame_to_txt(dev_bin5, "../data/%s/random_bins/dev/wfst/dev_bin5" % dataset, False)
    frame_to_txt(dev_bin4, "../data/%s/random_bins/dev/wfst/dev_bin4" % dataset, False)
    frame_to_txt(dev_bin3, "../data/%s/random_bins/dev/wfst/dev_bin3" % dataset, False)
    frame_to_txt(dev_bin2, "../data/%s/random_bins/dev/wfst/dev_bin2" % dataset, False)
    frame_to_txt(dev_bin1, "../data/%s/random_bins/dev/wfst/dev_bin1" % dataset, False)
    frame_to_txt(dev_bin6, "../data/%s/random_bins/dev/wfst/dev_bin6" % dataset, True)
    frame_to_txt(dev_bin5, "../data/%s/random_bins/dev/wfst/dev_bin5" % dataset, True)
    frame_to_txt(dev_bin4, "../data/%s/random_bins/dev/wfst/dev_bin4" % dataset, True)
    frame_to_txt(dev_bin3, "../data/%s/random_bins/dev/wfst/dev_bin3" % dataset, True)
    frame_to_txt(dev_bin2, "../data/%s/random_bins/dev/wfst/dev_bin2" % dataset, True)
    frame_to_txt(dev_bin1, "../data/%s/random_bins/dev/wfst/dev_bin1" % dataset, True)
    print("writing train_dev bins to .txt for wfsts...")
    frame_to_txt(train_dev_bin6, "../data/%s/random_bins/train_dev/wfst/train_dev_bin6" % dataset, False)
    frame_to_txt(train_dev_bin5, "../data/%s/random_bins/train_dev/wfst/train_dev_bin5" % dataset, False)
    frame_to_txt(train_dev_bin4, "../data/%s/random_bins/train_dev/wfst/train_dev_bin4" % dataset, False)
    frame_to_txt(train_dev_bin3, "../data/%s/random_bins/train_dev/wfst/train_dev_bin3" % dataset, False)
    frame_to_txt(train_dev_bin2, "../data/%s/random_bins/train_dev/wfst/train_dev_bin2" % dataset, False)
    frame_to_txt(train_dev_bin1, "../data/%s/random_bins/train_dev/wfst/train_dev_bin1" % dataset, False)
    frame_to_txt(train_dev_bin6, "../data/%s/random_bins/train_dev/wfst/train_dev_bin6" % dataset, True)
    frame_to_txt(train_dev_bin5, "../data/%s/random_bins/train_dev/wfst/train_dev_bin5" % dataset, True)
    frame_to_txt(train_dev_bin4, "../data/%s/random_bins/train_dev/wfst/train_dev_bin4" % dataset, True)
    frame_to_txt(train_dev_bin3, "../data/%s/random_bins/train_dev/wfst/train_dev_bin3" % dataset, True)
    frame_to_txt(train_dev_bin2, "../data/%s/random_bins/train_dev/wfst/train_dev_bin2" % dataset, True)
    frame_to_txt(train_dev_bin1, "../data/%s/random_bins/train_dev/wfst/train_dev_bin1" % dataset, True)
    print("writing test data")
    frame_to_txt(test, "../data/%s/random_bins/wfst_test/test" % dataset, False)


generate_bins("movies")
generate_bins("atis")
