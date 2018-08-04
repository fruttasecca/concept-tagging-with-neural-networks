#!/usr/bin/python3
import math  # for math.log
import os  # doing fst/opengrm commands
import sys  # arguments
import time # to timestamp working directories

from data_manager import Data

"""
File to run wfsts on atis and movies;
on atis it will result on an F1 of 93.08, on movies
it will be 82.74, to get to 82.96 the data in 
data/movies/wfst must be elaborated as in "elaboration3"
of https://github.com/fruttasecca/concept-tagging-with-WFST
"""


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


def run(train_data, concepts_phrases, test_data, gram, tech):
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
    """
    # get data
    data = Data(train_data)
    # needed for output, in case we substituted O's concepts with tokens or anything else
    class_set = set([concept for concept in data.counter_concepts if (concept[:2] == "B-" or concept[:2] == "I-")])
    class_set.add("O")

    timestamp = time.time()
    test_file = open(test_data, 'r')
    output_file = open("%s_order=%s_tech=%s_%f.txt" % (train_data.split("/")[-1], gram, tech, timestamp), "w")
    # if its not an absolute path make it into one since we are changing dir later
    if concepts_phrases[0] != "/":
        concepts_phrases = os.getcwd() + "/" + concepts_phrases

    # create dir, report problem if it exists already
    dir_name = "%s_%s_%s_%f" % (train_data.split("/")[-1], gram, tech, timestamp)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    os.chdir(dir_name)

    ############################
    ############################
    # create and train model
    # make word-concept transducer with -math.log as cost
    write_word_concept_transducer_same_prob(data)

    # make opengrm lexicon
    cmd = "ngramsymbols ../%s > train.syms" % train_data
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

            # write line to file (word label predicted_label)
            for word, label, predicted in zip(phrase, labels, predicted_labels):
                # these 2 checks are needed to clean out extra classes from data elaboration
                if predicted not in class_set:
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

    if len(sys.argv) != 6:
        print("usage: ./wfst.py train.data concept-sentences test.data gram smoothing")
        print("Train.data is a file in 1 word per line format, sentences are separated by an empty line; the first "
              "columns is tokens, the second columns is concepts")
        print("Concept-sentences is a file where each line contains a sentence only in the form of concepts, "
              "so every line is basically a list of concepts, which map to a phrase; i.e. if the first phrase in the "
              "train data would be mapped to 'O O B-movie.name' then the first sentence in concept sentences would be "
              "'O O B-movie.name'; see the atis or movies data in the data/<dataset>/wfst directory for an example.")
        print("Test.data is a file in 1 word per line format, sentences are separated by an empty line; the first "
              "columns is tokens, the second columns is concepts")
        print("While running, a temporary directory will be created, the directory and the output file are named in "
              "the following way: trainfile_gram_smoothing_timestamp.")

    elif sys.argv[4] not in grams:
        print("grams should be among the following:")
        print(grams)
    else:
        train_data = sys.argv[1]
        assert os.path.isfile(train_data), "train data is not there"

        concept_sentences = sys.argv[2]
        assert os.path.isfile(concept_sentences), "concept_sentences file is not there"

        test_data = sys.argv[3]
        assert os.path.isfile(test_data), "test data is not there"

        gram = sys.argv[4]
        assert gram in grams, "Grams should be among the following:\n%s" % grams

        tech = sys.argv[5]
        assert tech in smoothing_tech, "Smoothing should be among the following:\n%s" % smoothing_tech

        run(train_data, concept_sentences, test_data, gram, tech)
