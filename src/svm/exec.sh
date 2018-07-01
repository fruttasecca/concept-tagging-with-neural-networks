make CORPUS=exp.train.data MODEL=train  SVM_PARAM="-t1 -d3 -s1 -r1 -c1 -m 8000 " FEATURE="F:-4..4:0..0 F:-1..0:2..3 F:0..0:0.. T:-2..-1" train
yamcha -m train.model < exp.test.data > svm.txt


