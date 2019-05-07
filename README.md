:point_down::point_down::point_down::point_down::point_down::mega:  
If you run into any trouble/problem feel free to contact me
at jacopo.gobbi@studenti.unitn.it


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
  - neural models, requirements: pytorch (0.4)

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

```sh
./movies_run.py ../../data/movies/crf/train_dev.txt ../../data/movies/crf/test.txt
./movies_run.py ../../data/movies/crf/train_dev.txt ../../data/movies/crf/test.txt ../../data/movies/w2v_trimmed.pickle ../../data/movies/c2v_20.pickle
```
To run the CRF script either without embeddings or with embeddings. Train and test 
files are in a 1 word per line format, w2v and c2v pickles are pickles mapping words
to their index. Defining more features is done programmatically in the script, by
adding your features to a features dictionary mapping the feature name to whatever
you want as a value; i.e. by doing "features['word[3]'] = word[:3]" you would add
the prefix of the word as a feature for the word currently selected.
The way this works is because of pycrfsuite, while this may seem cumbersome at first
you should find defining features at the script level easier than rewriting train and 
test files as for YAMCHA.
Note that using embeddings might me very memory consuming, especially 
for ATIS.

```sh
./run_model.py  --hidden_size=200 --epochs=5 --batch=5 --drop=0.7  --embedding_norm=6.0 --lr=0.001 --model=conv  --unfreeze --write=results.txt --train="../data/movies/train.pickle" --test="../data/movies/dev.pickle" --w2v="../data/movies/w2v_trimmed.pickle" --hidden_size=200 --bidirectional --dev
```
This will run the CONV model with the defined paramaters and hyperparameters.
Train and test files are provided as pickles containing a "tokens" column and a "concepts" 
column, where every entry is a list of strings (sentence in tokens forms or sentence in concepts forms).
So if the first sentence is "hello there" mapped to "O O", the first entry of the
tokens column would contain ["hello", "there"], while the first for the concepts
column would contain ["O", "O"].
For a more complete explanation and default values of hyperparameters simply run:
```sh
./run_model.py --help
```



For scripts that require w2v embeddings (and permit c2v embeddings as an extra parameter),
you must provide those embeddings as a pickle containing a pandas dataframe
with two columns, a "token" column and a "vector" column containing the
embedding vector. Each entry in this dataframe is simply a token and its corresponding
embedding vector.

This allows you to not use the whole google embeddings bin file (which is quite larger), but to
just "trim" down the embeddings to the ones you need, only keeping the embeddings of tokens that
are present in your dataset. Pickles containing trimmed embeddings for atis
and movies are already there, in their respective data directories.




##### Update
The revised version of the repository contains the revised version of the paper and
the folds (10) used for significance testing, they can be found in the atis data directory
or in the movies data directory.

The folds have been obtained from the training data by using sklearn KFold, seeded with 1337;
to perform runs involving neural nets, pytorch has been seeded with 999 and set to deterministic.
To compute the Welch's t-test sklearn has been used.

Note that based
on the pytorch or CUDA version you might have some different results, this is not caused
by the code in this project but stems from how pytorch and CUDA currently interact.
