# Concept tagging with recurrent neural networks.
This is the repository for the second project of the Language  
Understanding System course of 2018 of unitn, it contains the report,  
 the results, some saved models and all the code.    
Implemented in pytorch, currently still have hardcoded cuda calls  
in it so cuda is a requirement. To better understand what the   
project is about refer to the report, this file is stricly pragmatical and how-to-use.  

Structure of the project:
![Alt text](/struct.png?raw=true "structure of the project")

## data
Here we have the data of the project and trained embeddings; NLSPARQL.* are the original files, while
other files (csvs or pickle or others) are used for easier handling of the data.  
Train.data and test.data are in a (word_token lemma pos IOB_concept) for each line format, with sentences
separated by a newline.  
Pre-trained google embeddings are of course not included in the repository.   
  

## data_analysis 
In this directory you can find plots and .txt files describing the data.
Upon entering the directory, there will be 2 files and 2 directories, the two files
are oov.txt and concept_lexicon_mismatch.txt, given their name it is quite obvious 
what kind of information they contain.  
The two directories are named train and test, their content is identical in   
the sense that they contain the same kind of charts and files (like a barchart   
describing the concept frequency), but related to different data.

## output
Directory containing results of the baseline, results for the best performing hyper params
for every architecture and results on the same hyper params but while not using bidirection, 
unfreezing embeddings during training, or both; best perfoming models for each architecture are 
also saved here.

## src
Here we have scripts, be sure to be in this directory when you run them   
so that files are correctly found or written to the proper directory.

---

```sh
$ ./w2v_trainer.py
```
To train w2v embeddings on train data, different embedding dimensions  
are possible.

---

---

```sh
$ ./trim_w2v.py
```
To trim w2v google embeddings down to only the words
present in our data.

---

---

```sh
$ ./c2v.py
```
This will allow you to obtain character embeddings starting from  
either your embeddings or google ones.

---

---

```sh
$ ./data_analysis.py
```
Runs data analysis scripts, saves plots and text files in the  
data_analysis directory of the project.

---

---

```sh
$ ./data_manager.py
```
Data management utility classes.

---

---

```sh
$ ./run_model.py
```
Allows you to train a model either on train data or train and dev, 
save the model, etc.

---

The src directory also contains a models directory, which
is where source code for the different architectures is located, you will find
8 models there, all based on recurrent neural networks.  
There is also a baseline directory, which contains scripts for baselines results.  

All scripts have an --help option to inform you about what parameters you
can use.

