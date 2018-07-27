make CORPUS=exp.train.data MODEL=train  SVM_PARAM="-t1 -d2 -s1.868520 -r1.964100 -c1.421407 -m 8000 " FEATURE="F:-4..4:0..0 F:-1..0:2..3 F:0..0:0.. T:-2..-1" train
yamcha -m train.model < exp.test.data | expand -t 1 > svm.txt
rm train*
