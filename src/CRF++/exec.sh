#!/bin/sh
./CRF++-0.58/crf_learn -c 20.0 -m 200 -p 8 template exp.train.data model
./CRF++-0.58/crf_test -m model exp.test.data > crf++.txt

#./CRF++-0.58/crf_learn -a MIRA template train.data model
#./CRF++-0.58/crf_test -m model test.data

#../../crf_learn -a CRF-L1 template train.data model
#../../crf_test -m model test.data

rm -f model
