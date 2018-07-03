#!/usr/bin/python3
import matplotlib.pyplot as plt  # drawing charts
import operator  # sorting dicts
import os
import pandas as pd

from data_manager import Data, w2v_matrix_vocab_generator


def plot_frequency(counter_dict, name='', firstn=None, bar_color="green"):
    """
    Given a dictionary mapping keys to values (usually a string to a frequency), plots a bar chart in decreasing
    order and saves it in the data_analysis directory.
    :param counter_dict: Dictionary containing words (key) and their frequency.
    :param name: Name to use a a part of the file name and other elements of the plot,
    the file will be named as name + " " + Frequency Chart.png.
    :param firstn: If only the firstn elements are to be used, taken in decreasing order of the value, helps in
    plotting due to using the words for the bar names.
    :param bar_color: Color of the bars of the plot, default is green.
    """
    # sort dict elements by value, most frequent first
    sorted_dw = sorted(counter_dict.items(), key=operator.itemgetter(1), reverse=True)
    words = [pair[0] for pair in sorted_dw]
    values = [pair[1] for pair in sorted_dw]
    # truncate if asked to
    if firstn is not None:
        words = words[:firstn]
        values = values[:firstn]

    plt.bar(range(len(words)), values, alpha=0.4, color=bar_color)
    # legend and bar names
    # title
    font = {'weight': 'bold',
            'size': 5}
    plt.rc('font', **font)
    # bar ticks
    plt.xticks(range(len(words)), words, rotation='vertical')
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=8)

    # rest of labels
    plt.xlabel(name, fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    # plt.title(name + ' ' + 'Frequency Chart')

    # keep labels into picture
    plt.tight_layout()

    plt.savefig("../data_analysis/%s frequency chart.png" % name)
    plt.clf()


def write_frequency(counter_dict, name='', firstn=None):
    """
    Given a dictionary mapping keys to values (usually a string to a frequency), writes word-frequency-% with respect to
    total to a file, saving it in the data_analysis directory.
    :param counter_dict: Dictionary containing words (key) and their frequency.
    :param name: Name to use a a part of the file, the file will be named as name + " " + frequencies.txt.
    :param firstn: If only the firstn elements are to be used, taken in decreasing order of the value.
    """
    # sort dict elements by value, most frequent first
    sorted_dw = sorted(counter_dict.items(), key=operator.itemgetter(1), reverse=True)
    words = [pair[0] for pair in sorted_dw]
    values = [pair[1] for pair in sorted_dw]
    # truncate if asked to
    if firstn is not None:
        words = words[:firstn]
        values = values[:firstn]
    total = sum(values)
    with open("../data_analysis/%s frequencies.txt" % (name), 'w') as file:
        for w, v in zip(words, values):
            file.write("%s %s %f\n" % (w, v, v / total))
        file.close()


def oov():
    """
    Compute oov rate and oov words, save results to data_analysis directory.
    Each oov word will have the concept to which it was mapped on its right.
    """
    train = Data("../data/train.data")
    test = Data("../data/test.data")

    train_lex = set(train.lexicon_words)
    test_lex = set(test.lexicon_words)
    oov = test_lex.difference(train_lex)
    concept_counter = dict()

    with open("../data_analysis/oov.txt", "w") as out:
        out.write("OOV rate:  %s\n" % (len(oov) / len(test_lex)))
        out.write("OOV WORDS:\n")
        for word in sorted(oov):
            for concept in test.lexicon_concepts:
                tmp = ""
                if test.word_concept(word, concept) > 0:
                    tmp = tmp + " " + concept
                    concept_counter[concept] = concept_counter.get(concept, 0) + 1
            out.write("%s %s\n" % (word, tmp))
        sorted_c = sorted(concept_counter.items(), key=operator.itemgetter(1))
        out.write("Concepts in order of oov times:\n")
        for k, v in reversed(sorted_c):
            out.write("%s %s\n" % (k, v))


def concept_lexicon_mismatch():
    """
    Checks on the train and test concept lexicons (no IOB notation) for concept that are in one set but not
    in the other, and writes results to data_analysis/concept_lexicon_mismatch.txt.
    """
    train_lex = set(Data("../data/train.data").lexicon_clean_concepts)
    test_lex = set(Data("../data/test.data").lexicon_clean_concepts)

    not_in_train = set()
    not_in_test = set()

    # get concepts that are in train but not in test
    for concept in train_lex:
        if concept not in test_lex:
            not_in_test.add(concept)

    # get concepts that are in test but not in train
    for concept in test_lex:
        if concept not in train_lex:
            not_in_train.add(concept)
    with open("../data_analysis/concept_lexicon_mismatch.txt", "w") as out:
        out.write("These concepts are part of the train data but not part of the test data:\n")
        for concept in not_in_test:
            out.write(concept + "\n")
        out.write("These concepts are part of the test data but not part of the train data:\n")
        for concept in not_in_train:
            out.write(concept + "\n")


def summaries(data, output_file):
    """
    Computes some statistics about the data and saves them as a file.
    :param data: Data for which to compute statistics, must be in
    (word lemma pos IOB) format.
    :param output_file: Where to write the statistics.
    """
    data = Data(data)
    number_of_phrases = len(data._data)
    size_word_lexicon = len(data.lexicon_words)
    size_lemma_lexicon = len(data.lexicon_lemmas)
    size_pos_lexicon = len(data.lexicon_pos)
    size_concept_lexicon = len(data.lexicon_concepts)
    size_concept_lexicon_no_iob = len(data.lexicon_clean_concepts)

    # get average length of phrases
    avg_length = 0
    for phrase in data._data:
        avg_length += len(phrase)
    avg_length /= len(data._data)

    with open(output_file, "w") as out:
        out.write("total phrases: %s\n" % number_of_phrases)
        out.write("average length of phrases: %s\n" % avg_length)
        out.write("word lexicon size: %s\n" % size_word_lexicon)
        out.write("lemma lexicon size: %s\n" % size_lemma_lexicon)
        out.write("pos lexicon size: %s\n" % size_pos_lexicon)
        out.write("concept lexicon size (with IOB): %s\n" % size_concept_lexicon)
        out.write("concept lexicon size (without IOB): %s\n" % size_concept_lexicon_no_iob)


def plot_data(data, output_dir, color):
    """
    Plot various frequencies (tokens, lemmas, etc.) of a given dataset in the form
    of (word, lemma, pos, concept) for each line, with phrases separated by an empty space.
    :param data: Data for which to plot and write different frequencies.
    :param output_dir: Output dir to which output files are going to be moved to.
    :param color: Color of the bars in the charts.
    """

    data = Data(data)

    ###
    # plot and write (.txt files) singletons counters
    c_words = data.counter_words
    c_lemmas = data.counter_lemmas
    c_pos = data.counter_pos
    c_concepts = data.counter_concepts
    c_clean_concepts = data.counter_clean_concepts
    plot_frequency(c_words, "Word", 50, color)
    write_frequency(c_words, "Word", 50)
    plot_frequency(c_lemmas, "Lemma", 50, color)
    write_frequency(c_lemmas, "Lemma", 50)
    plot_frequency(c_pos, "Pos tag", 50, color)
    write_frequency(c_pos, "Pos tag", 50)
    plot_frequency(c_concepts, "Concept", 50, color)
    write_frequency(c_concepts, "Concept", 50)
    c_concepts.pop('O')
    plot_frequency(c_concepts, "Concept (no O)", 50, color)
    write_frequency(c_concepts, "Concept (no O)", 50)
    c_clean_concepts.pop("O")
    plot_frequency(c_clean_concepts, "Concepts (no O no iob)", 50, color)
    write_frequency(c_clean_concepts, "Concepts (no O no iob)", 50)

    ###
    # plot and write (.txt files) pairs counters
    cp_word_concept = data.counter_word_concept
    cp_lemma_concept = data.counter_lemma_concept
    cp_pos_concept = data.counter_pos_concept
    # remove "O" pairs
    for key, v in list(cp_word_concept.items()):
        if key[-2:] == " O":
            cp_word_concept.pop(key)
    for key, v in list(cp_lemma_concept.items()):
        if key[-2:] == " O":
            cp_lemma_concept.pop(key)
    for key, v in list(cp_pos_concept.items()):
        if key[-2:] == " O":
            cp_pos_concept.pop(key)
    plot_frequency(cp_word_concept, "word-concept", 50, color)
    write_frequency(cp_word_concept, "word-concept", 50)
    plot_frequency(cp_lemma_concept, "lemma-concept", 50, color)
    write_frequency(cp_lemma_concept, "lemma-concept", 50)
    plot_frequency(cp_pos_concept, "pos tag-concept", 50, color)
    write_frequency(cp_pos_concept, "pos tag-concept", 50)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system("mv ../data_analysis/*.txt %s " % output_dir)
    os.system("mv ../data_analysis/*.png %s " % output_dir)


def embeddings_oov(df_data, path):
    """
    Data analysis on words and lemmas missing an embedding.

    :param df_data: Dataframe of sentences.
    :param path: Where to save results.
    """
    w2v_vocab, _ = w2v_matrix_vocab_generator("../data/w2v_trimmed.pickle")
    c2v_vocab, _ = w2v_matrix_vocab_generator("../data/c2v_300.pickle")
    missing_tokens = dict()
    missing_lemmas = dict()
    for _, row in df_data.iterrows():
        for token in row["tokens"]:
            if token.isdigit():
                token = "number"
            if token not in w2v_vocab and token.title() not in w2v_vocab:
                missing_tokens[token] = missing_tokens.get(token, 0) + 1
        for lemma in row["lemmas"]:
            if lemma not in w2v_vocab and lemma.title() not in w2v_vocab:
                missing_lemmas[lemma] = missing_lemmas.get(lemma, 0) + 1
    sorted_tokens = sorted(missing_tokens.items(), key=operator.itemgetter(1), reverse=True)
    sorted_lemmas = sorted(missing_lemmas.items(), key=operator.itemgetter(1), reverse=True)

    with open(path, "w") as out:
        out.write("missing tokens: %i\n" % len(missing_tokens))
        out.write("missing lemmas: %i\n" % len(missing_lemmas))
        out.write("TOKEN : COUNT\n")
        for token, count in sorted_tokens:
            out.write("%s %i\n" % (token, count))
        out.write("LEMMA : COUNT\n")
        for lemma, count in sorted_lemmas:
            out.write("%s %i\n" % (lemma, count))







# clean up
os.system("rm -rf ../data_analysis/*")

# plot frequencies of train and test data
plot_data("../data/train.data", "../data_analysis/train/", "green")
plot_data("../data/test.data", "../data_analysis/test/", "blue")

# do oov stats
oov()
concept_lexicon_mismatch()

# summaries for train and test data
summaries("../data/train.data", "../data_analysis/train/summary.txt")
summaries("../data/test.data", "../data_analysis/test/summary.txt")

# check for words missing embeddings
df_tmp1 = pd.read_pickle("../data/train.pickle")
df_tmp2 = pd.read_pickle("../data/dev.pickle")
df_train = pd.concat((df_tmp1, df_tmp2))
df_test = pd.read_pickle("../data/test.pickle")
embeddings_oov(df_train, "../data_analysis/train/embeddings.txt")
embeddings_oov(df_test, "../data_analysis/test/embeddings.txt")
