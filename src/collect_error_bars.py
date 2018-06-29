#!/usr/bin/python3
import os
import operator
import numpy as np


def collect_results():
    """
    Collects the results in a directory (output files, i.e. random.txt), writing them in a file
    named f1.scores in the work directory.
    """
    # list of strings of the form "parameters: f1"
    res = dict()
    dir_name = "f1/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # measure performance of each file and retrieve f1 score, writing it to a file in the f1 directory
    for file in os.listdir("."):
        if file.endswith(".txt"):
            input_name = file
            output_name = dir_name + file + "_res.txt"
            cmd = "./conlleval.pl < %s | sed '2q;d' | grep -oE '[^ ]+$' > %s" % (input_name, output_name)
            os.system(cmd)

    # collect different f1 scores in the same file
    for file in sorted(os.listdir("f1")):
        if file.endswith(".txt"):
            with open("f1/" + file, "r") as input:
                name = file.split(".")[0]
                for line in input:
                    res[name] = float(line.strip("\n"))

    values = np.array([v for v in res.values()])
    mean, std = values.mean(), values.std()
    res = sorted(res.items(), key=operator.itemgetter(1), reverse=True)
    with open("f1.scores", "w") as output:
        output.write("MEAN: %f, STD: %f\n" % (mean, std))
        for pair in res:
            output.write(pair[0] + " " + str(pair[1]) + "\n")
        output.close()

    os.system("rm -r f1")


# get to directory and for each sub directory collect results
os.chdir("../output/error_bars")
cwd = os.getcwd()
dirs = [cwd + "/" + x[0] for x in os.walk(".")]
for dir in dirs:
    if dir[-1] != ".":
        print(dir)
        os.chdir(dir)
        collect_results()
