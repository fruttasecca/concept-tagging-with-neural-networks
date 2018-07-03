#!/usr/bin/python3

# quick and dirty script to rewrite data for CRF++
def expand(file):
    with open(file, "r") as input:
        with open("exp." + file, "w") as output:
            for line in input:
                tokenize = line.split()
                if len(tokenize) == 0:
                    output.write("\n")
                else:
                    word = tokenize[0]
                    # prefix and postfix
                    pre = word[:3] if len(word) > 2 else word
                    post = word[-3:] if len(word) > 2 else word
                    tokenize.insert(-1, pre)
                    tokenize.insert(-1, post)
                    tokenize.insert(-1, "1")
                    output.write(" ".join(tokenize) + "\n")


expand("train.data")
expand("test.data")
