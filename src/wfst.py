#!/usr/bin/python3
import math  # for math.log
import os  # doing fst/opengrm commands
import sys  # arguments

from data_manager import Data
from data_manager import class_vocab_movies, class_vocab_atis


def write_word_concept_transducer_same_prob(data):
    """
    Write the word to concept transducer as a .txt file of transitions, to be later compiled.
    This function will make it so that <unk> words will have an equal probability of being mapped to all the different
    concepts.
    :param data: Data class containing infos on the corpus.
    """
    wt_file = open('word_concept.txt', 'w')

    # word-tag costs based on counts
    for pair in data.counter_word_concept:
        word, concept = pair.split()
        cost = -math.log(data.counter_word_concept[pair] / data.counter_concepts[concept])
        wt_file.write("0 0 %s %s %f\n" % (word, concept, cost))

    # <UNK>-tag maps with uniform cost with every concept
    for concept in data.lexicon_concepts:
        wt_file.write("0 0 <unk> %s %f\n" % (concept, 1. / len(data.lexicon_concepts)))
    wt_file.write("0\n")
    wt_file.close()


def run(train_data, concepts_phrases, test_data, gram, tech, dataset):
    """
    Trains the model given gram length and smoothing, then runs it against the test data, an output file
    will be written in the output directory of this project, the file is going to be named as
    name of this script_gram_tech, i.e.:
    v1_order=2_tech=kneser_ney.txt

    :param train_data: Training data in (word lemma pos IOB) format for each line, with phrases separated by a newline.
    :param concepts_phrases: Training concept phrases, each line contains a phrase, used for ngramcounting to make
    the prior.
    :param test_data: Test data in (word lemma pos IOB) format for each line, with phrases separated by a newline.
    :param gram: Length of the gram used, --order=gram is going to be used in ngramcount.
    :param tech: Smoothing technique, --method=tech is going to be used in ngrammake.
    :param cut_off: Which kind of cut_off to use : empty string for no cut_off, thres for counting out words
    under a certain thres, and distr for counting out words under a certain thres and distributing the probability
    of mapping <unk> to a concept in a way proportional to how many word-concept pairs have been removed this way.
    """
    # get data
    data = Data(train_data)
    test_file = open(test_data, 'r')
    output_file = open("%s_%s_order=%s_tech=%s.txt" % (dataset, train_data.split("/")[-1], gram, tech), "w")
    # if its not an absolute path make it into one since we are changing dir later
    if concepts_phrases[0] != "/":
        concepts_phrases = os.getcwd() + "/" + concepts_phrases

    # create dir, report problem if it exists already
    dir_name = "tmp_%s_%s_%s" % (train_data.split("/")[-1], gram, tech)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        print("there is already a working directory named %s, perhaps a process with the same arguments is already "
              "running" % dir_name)
        exit()
    os.chdir(dir_name)


    ############################
    ############################
    # create and train model
    # make word-concept transducer with -math.log as cost
    write_word_concept_transducer_same_prob(data)

    # make opengrm lexicon
    if dataset == "movies":
        cmd = "ngramsymbols ../../data/%s/train.data > train.syms" % dataset
    elif dataset == "atis":
        cmd = "ngramsymbols ../../data/%s/random_bins/train_dev/wfst/train_dev_bin6.txt > train.syms" % dataset
    os.system(cmd)

    # compile word to concept fst
    cmd = "fstcompile --isymbols=train.syms --osymbols=train.syms word_concept.txt > word_concept.fst"
    os.system(cmd)

    # write concepts phrases as a far archive
    cmd = "farcompilestrings --symbols=train.syms --unknown_symbol='<unk>' %s > train.concepts.far " % concepts_phrases
    os.system(cmd)

    # make tag acceptor
    cmd = "ngramcount --order=%s --require_symbols=false train.concepts.far > train.concepts.cnts;" % gram
    os.system(cmd)
    cmd = "ngrammake --method=%s train.concepts.cnts > concepts.fsa" % tech
    os.system(cmd)


    ###########################
    ###########################
    # run model on test data
    phrase = []
    labels = []  # label for each word
    for line in test_file:
        split = line.split()
        if len(split) > 0:
            # keep building current phrase
            if split[0] in data.counter_words:
                phrase.append(split[0])
                labels.append(split[-1])
            else:
                # word is unknown if was not part of train lexicon
                phrase.append("<unk>")
                labels.append(split[-1])
        else:
            # make acceptor out of phrase
            cmd = 'echo "%s" | farcompilestrings --symbols=train.syms --unknown_symbol="<unk>" --generate_keys=1 ' \
                  '--keep_symbols | farextract --filename_suffix=".fst"' % (" ".join(phrase))
            os.system(cmd)

            # compose phrase with word_concept transducer
            cmd = "fstcompose 1.fst word_concept.fst > 2.fst"
            os.system(cmd)

            # compose the obtained fst with the concept sentences acceptor
            cmd = "fstcompose 2.fst concepts.fsa | fstrmepsilon | fstshortestpath | fstrmepsilon | fsttopsort | " \
                  "fstprint --isymbols=train.syms --osymbols=train.syms > res.info"
            os.system(cmd)

            # read res.info to get the predicted labels
            res = open("res.info", "r")
            predicted_labels = []
            for res_line in res:
                split = res_line.split()
                if len(split) == 5:  # needed because last line is the final state cost
                    predicted_labels.append(split[3])
            res.close()

            if dataset == "movies":
                class_vocab = class_vocab_movies
            elif dataset == "atis":
                class_vocab = class_vocab_atis

            # write line to file (word label predicted_label)
            for word, label, predicted in zip(phrase, labels, predicted_labels):
                # these 2 checks are needed to clean out extra classes from data elaboration
                if predicted not in class_vocab:
                    predicted = "O"
                output_file.write("%s %s %s\n" % (word, label, predicted))
            output_file.write("\n")

            phrase = []
            labels = []

    output_file.close()
    test_file.close()

    ############################
    ############################
    # clean up
    os.chdir("..")
    os.system("rm -r %s" % dir_name)


if __name__ == "__main__":
    # smoothing techniques to use, all of them are going to be used, differently named outputs will be created
    smoothing_tech = ["witten_bell", "absolute", "kneser_ney", "presmoothed", "unsmoothed", "katz"]
    # length of grams
    grams = ["1", "2", "3", "4", "5"]

    if len(sys.argv) != 6 and len(sys.argv) != 7:
        print("usage: ./train_and_test.py train_data concept_phrases test_data gram-length smoothing-method")
    elif sys.argv[4] not in grams:
        print("grams should be among the following:")
        print(grams)
    elif sys.argv[5] not in smoothing_tech:
        print("smoothing should be among the following:")
        print(smoothing_tech)
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
