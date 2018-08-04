# Concept tagging with WFSTs, SVMs, CRFs, and neural networks
This is the repository for the paper "Concept Tagging for Natural Language Understanding: Two Decadelong Algorithm Development", http://arxiv.org/abs/1807.10661.
Please refer to this paper when using this code for your papers or reports.
It started as a course project for the course of Language Understanding Systems (2017-2018), part of the master course of computer science of the University of Trento.
The course is held by Professor Giuseppe Riccardi, the material can be found at http://disi.unitn.it/~riccardi/page7/page13/page13.html.

In this project we tried out concept tagging on two datasets, ATIS and NL-SPARQL, with many different
architectures, the aim of the project was not to obtain the highest possible F1 score but to test the same task
under many different models, taking into consideration the number of parameters, ease of setup, etc.


For each type of model, we have tried to keep things easy and concise, so that anyone can either
modify them and run on these datasets, or just copy the specific model they need and repurpose it for their own
task.
Most of the scripts are really simple, once they have been run they should either work or tell you
what arguments you are currently not providing, and what you should/could provide.

# Architectures:
    
  - WFSTs, requirements: opengrm, openfst
  - SVMs, requirements: YAMCHA
  - CRFs, requirements: pycrfsuite
  - neural models, requirements: pytorch

![Alt text](/struct.png?raw=true "structure of the repository")

The repository is organized in this way: 

In src you will find:
  - data_manager.py, which contain classes for managing or transforming data
  - collect_results.py, a utility script to compute the F1 score of many result files a once
  - models, directory containing the source code of the different nn models
  - run_model.py, to train, test, save models and their results
  - pycrfsuite, directory containing scripts to run crfs (1 for atis, 1 for movies)
  - svm, directory containing an atis and movies directories, which have scripts
  to run svms (YAMCHA) on either atis or movies
  - wfst.py, script to run WFST
  
In data you will find two directories, one named atis and the other movies, here
data is stored, more specifically, for each dataset:
  - at the first level we have the original data, pickles used by nn, w2v and c2v embeddings
  - a crf directory containing data in a format to be used by the crf scripts
  - a wfst directory containing data in a format to be used by the wfst scripts
  - a svm directory containing data in a format to be used by the svm scripts
  - data formatted in a way to be used by svms is currently in src/dataset/svm, will be moved in a following commit
  
 
The output directory contains:
  - a directory named atis, containing results from wfsts, svms, crfs, nn (error bars) for the atis dataset
  - a directory named movies, containing results from wfsts, svms, crfs, nn (error bars) for the movies dataset
  - the conlleval.pl evaluation script, used to evaluate performance of results in the two aforementioned directories
  
 
YAMCHA, SVM and the CRF scripts run using files formatted in a 1 word per line format,
while the nn scripts use pickles from pandas dataframes. 
A dataframe must contain columns "tokens" and "concepts", the entries of each sample/row
are list of strings, meaning that, given sentence zero being "hi there", the first row
of the dataframe will have the "tokens" entry equal as ["hi", "there"] and the "concepts" entry
equal to ["0", "0"] (or whatever concepts they are mapped to).


Examples on how to call the scripts for the different architectures:

```sh
./wfst.py ../data/atis/wfst/train.txt ../data/atis/wfst/concept_sentences.txt ../data/atis/wfst/test.txt 4 kneser_ney 
```
To run the WFST script, train and test are in 1 word per line format, concept_sentences have
one sentence of concepts per line, so given a sentence "hi there" mapped to "O O ", the first
entry of this file would just be "O O". If you have any doubts check the files in data/<dataset>/wfst.

```sh
./exec.sh ../../../data/movies/svm/exp.train.txt ../../../data/movies/svm/exp.test.txt
```
To run YAMCHA (SVMs) on these files, training on the first and testing on the second. To add
your features you should add the feature as a column in those files, and edit the exec.sh
file to make YAMCHA use those features. The way this works is because of YAMCHA; check its
documentation if you have doubts.



