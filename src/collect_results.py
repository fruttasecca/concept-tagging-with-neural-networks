#!/usr/bin/python3
import os
import sys
import operator
import numpy as np

"""
Script to collect f1 scores of different output files that are in the provided directory, the script conlleval.pl in the
output directory is used; the provided directory will afterwards contain a file named f1.scores, containing the sorted
scores of the different files, their mean and std.
"""


def collect_results(dir):
    """
    Collects the results in a directory (output files, i.e. random.txt), writing them in a file
    named f1.scores in the same directory.
    """
    dir = dir.rstrip("/") + "/" # make sure the / is there without having two of them
    f1dir = dir + "f1/"
    if not os.path.exists(f1dir):
        os.makedirs(f1dir)

    # list of strings of the form "parameters: f1"
    res = dict()

    # measure performance of each file and retrieve f1 score, writing it to a file in the f1 directory
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            input_name = dir + file
            output_name = f1dir + file
            cmd = "../output/conlleval.pl < %s | sed '2q;d' | grep -oE '[^ ]+$' > %s" % (input_name, output_name)
            os.system(cmd)

    # collect different f1 scores in the same file
    for file in sorted(os.listdir(f1dir)):
        if file.endswith(".txt"):
            with open(f1dir + file, "r") as input:
                name = file
                for line in input:
                    res[name] = float(line.strip("\n"))

    values = np.array([v for v in res.values()])
    mean, std = values.mean(), values.std()
    res = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
    with open(dir + "f1.scores", "w") as output:
        output.write("MEAN: %f, STD: %f\n" % (mean, std))
        for pair in res:
            output.write(pair[0] + " " + str(pair[1]) + "\n")
        output.close()

    os.system("rm -r %s" % f1dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: \n./collect_results path \nWhere path is a directory containing .txt files with predictions; "
              "this script will collect the F1 score of each prediction file and report them in a file called "
              "f1.scores; the evaluation script conlleval.pl in the output directory is used.")
        exit()
    else:
        # get to directory and for each sub directory collect results
        dir = sys.argv[1]
        collect_results(dir)
