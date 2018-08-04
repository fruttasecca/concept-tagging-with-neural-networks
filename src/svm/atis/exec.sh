make CORPUS=$1 MODEL=train  SVM_PARAM="-t1 -d1 -s1.235739 -r1.444492 -c0.401410 -m 8000" FEATURE="F:-6..4:0..0 F:-1..0:1..2 F:0..0:0.. T:-2..-1" train
yamcha -m train.model < $2 | expand -t 1 > svm.txt
rm train*


