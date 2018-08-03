import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class PytorchDataset(Dataset):
    """Dataset to import augmented data."""

    def __init__(self, df, init_transform, getitem_transform=None):
        """
        :param df Dataframe containing data, must have "concepts" and "tokens" columns, every entry of such a column
        is a list of strings, entries for the same sample must have the same length, given that they are representing
        the same sentence, either using tokens or concepts.
        :param init_transform: Transform function to be used on data points at import time.
        :param getitem_transform: Transform function to be used on data points when they are retrieved with __getitem__.
        """

        self.init_transform = init_transform
        self.getitem_transform = getitem_transform

        # transform and save data
        self.data = dict()
        for i in range(len(df)):
            sample = df.iloc[i, :]
            self.data[i] = self.init_transform(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.getitem_transform(self.data[idx]) if self.getitem_transform is not None else self.data[idx]


def w2v_matrix_vocab_generator(w2v_pickle):
    """
    Creates the w2v dict mapping word to index and a numpy matrix of (num words, size of embedding), words will
    be mapped to their index, such that row ith will be the embedding of the word mapped to the i index.

    :param w2v_pickle: Dataframe containing token and vector columns, where token is a string and vector is the
    embedding vector, each vector must have the same length (and the length must be equal to the argument embedding_dim).
    :return: A dict, np matrix pair, the dict maps words to indexes, the matrix ith row will contain the embedding
    of the word mapped to the ith index.
    """
    # create internal w2v dictionary
    w2v_df = pd.read_pickle(w2v_pickle)
    w2v = dict()
    embedding_dim = len(w2v_df.iloc[0, 1])

    # shape +2 for unknown and padding tokens
    w2v_weights = np.zeros(shape=(w2v_df.shape[0] + 2, embedding_dim))
    for index, data_point in w2v_df.iterrows():
        curr_index = len(w2v)
        w2v[data_point["token"]] = curr_index
        w2v_weights[curr_index, :] = np.array(data_point["vector"])
    w2v["<UNK>"] = len(w2v_weights) - 2
    w2v["<padding>"] = len(w2v_weights) - 1
    return w2v, w2v_weights


class Data(object):
    """
    Class to contain data, once initialized it is organized in this way:
    - properties that allow you to get counter maps for words, lemmas, pos tags, concepts, and pairs
    - properties that allow you to get words, lemmas, pos tags, concepts lexicon
    - methods that allow you to get the counter for a specific word, lemma, pos tags, concept or pair
    note: this class is something i copy pasted from an old project of mine, not really that useful in here, mostly used
    for the wfst script.
    """

    def __init__(self, file):
        """
        Inits the structure, importing the file
        and elaborating all the counts.
        :param file: File to pass, format should be
        token lemma pos concept for each line, separated
        by an empty line to signal the end of a phrase.
        """
        self._data = []  # list of phrases, each phrase is a list of data points (word lemma pos concept)
        with open(file, 'r') as file:
            phrase = []
            for line in file:
                split = line.split()
                if len(split) > 0:
                    # keep building current phrase
                    phrase.append(split)
                else:
                    # end of phrase, append it to data
                    self._data.append(phrase)
                    phrase = []

        """
        init and compute counters
        """
        # singletons
        self.__words_counter = dict()
        self.__concepts_counter = dict()
        self.__concepts_clean_counter = dict()  # concepts without IOB notation
        # pairs of stuff
        self.__word_concept_counter = dict()

        for phrase in self._data:
            for data_point in phrase:
                word, concept = data_point
                # singletons
                self.__words_counter[word] = 1 + self.__words_counter.get(word, 0)
                self.__concepts_counter[concept] = 1 + self.__concepts_counter.get(concept, 0)
                clean_c = concept if concept == "O" else concept[2:]
                self.__concepts_clean_counter[clean_c] = 1 + self.__concepts_clean_counter.get(clean_c, 0)
                # pairs of stuff
                self.__word_concept_counter[word + " " + concept] = 1 + self.__word_concept_counter.get(
                    word + " " + concept, 0)

    @property
    def size(self):
        """
        :return: Number of phrases stored.
        """
        return len(self._data)

    @property
    def counter_words(self):
        """
        :return: Dictionary that maps a word to its counter.
        """
        return self.__words_counter

    @property
    def counter_concepts(self):
        """
        :return: Dictionary that maps a concept to its counter.
        """
        return self.__concepts_counter

    @property
    def counter_clean_concepts(self):
        """
        :return: Dictionary that maps a concept (no IOB) to its counter.
        """
        return self.__concepts_clean_counter

    @property
    def counter_word_concept(self):
        """
        :return: Dictionary that maps a word + concept pair to its counter, separated by space.
        """
        return self.__word_concept_counter

    @property
    def lexicon_words(self):
        """
        :return: List of words in the corpus.<epsilon> and <unk> not included.
        """
        return list(self.counter_words.keys())

    @property
    def lexicon_concepts(self):
        """
        :return: List of concepts in the corpus.<epsilon> and <unk> not included.
        """
        return list(self.counter_concepts.keys())

    @property
    def lexicon_clean_concepts(self):
        """
        :return: List of clean concepts  (no IOB notation ) in the corpus.<epsilon> and <unk> not included.
        """
        return list(self.__concepts_clean_counter.keys())

    def word(self, word):
        """
        :param word: Word for which to return the count for.
        :return: Count of the word, >= 0.
        """
        return self.__words_counter.get(word, 0)

    def concept(self, concept):
        """
        :param concept: Concept for which to return the count for.
        :return: Count of the concept, >= 0.
        """
        return self.__concepts_counter.get(concept, 0)

    def word_concept(self, word, concept):
        """
        :param word: Word of the word - concept pair.
        :param concept: Concept of the word - concept pair.
        :return: Count of the word - concept pair >= 0.
        """
        return self.__word_concept_counter.get(word + " " + concept, 0)

    def to_dataframe(self):
        """
        Transform the data to a df containing the tokens, lemmas, pos and concepts columns.
        Each sentence will correspond to a row, each column (for each row) contains a list of strings.
        :return: Dataframe transposition of this data object.
        """
        df = pd.DataFrame(columns=["tokens", "concepts"])
        for i, phrase in enumerate(self._data):
            words = []
            concepts = []
            for data_point in phrase:
                word, concept = data_point
                words.append(word)
                concepts.append(concept)
            df.loc[i] = [words, concepts]
        return df


def batch_sequence(batch, device):
    """
    Given a batch return sequence data, labels and chars data (if present) in a "batched" way, as a tensor
    where the first dimension is the dimension of the batch.
    :param batch: List of sample points, each sample point should contain "tokens" and "concepts" data, list
    of integers, it may also contain "chars" data, which is a list of integers.
    """
    list_of_data_tensors = [sample["tokens"].unsqueeze(0) for sample in batch]
    data = torch.cat(list_of_data_tensors, dim=0)
    list_of_labels_tensors = [sample["concepts"].unsqueeze(0) for sample in batch]
    labels = torch.cat(list_of_labels_tensors, dim=0)
    char_data = None
    if "chars" in batch[0]:
        list_of_char_data_tensors = [sample["chars"] for sample in batch]
        char_data = torch.cat(list_of_char_data_tensors, dim=0).to(device)
    return data.to(device), labels.to(device), char_data


class DropTransform(object):
    """ Transformer class to be passed to the pytorch dataset class to transform data at run time, it randomly
    drops word indexes to 'simulate' unknown words."""

    def __init__(self, drop_chance, unk_idx, preserve_idx):
        """
        :param drop_chance: Chance of dropping a word.
        :param unk_idx: Which index to use in place of the one of the dropped word.
        :param preserve_idx: Index to never drop (i.e. the padding index).
        """
        self.drop_chance = drop_chance
        self.unk_idx = unk_idx
        self.preserve_idx = preserve_idx

    def _might_drop(self, idx):
        """
        Drop idx by chance and if its not the index to preserve.
        :param idx:
        :return:
        """
        return self.unk_idx if (random.uniform(0, 1) < self.drop_chance and idx != self.preserve_idx) else idx

    def __call__(self, sample):
        """
        Get a sample, concepts and char embeddings idxs (if present) are preserved, each token is instead
        replaced by a chance equal to self._drop_chance.

        :param sample:
        :return:
        """
        tsample = dict()
        seq = sample["tokens"].clone()
        for i in range(len(seq)):
            seq[i] = self._might_drop(sample["tokens"][i].item())
        tsample["tokens"] = seq
        tsample["concepts"] = sample["concepts"]
        tsample["sequence_extra"] = sample["sequence_extra"]
        if "chars" in sample:
            tsample["chars"] = sample["chars"]
        return tsample


class InitTransform(object):
    """ Transformer class to be passed to the PytorchDataset class to transform data at import time, given a sample,
    returns a transformed sample, which is a dict mapping keys to tensors of either w2v indexes, c2v indexes, concept
    indexes, see this class __call__ method for more info.
    """

    def __init__(self, w2v_vocab, class_vocab, c2v_vocab=None, sentence_length_cap=50, word_length_cap=30,
                 add_matrix=True):
        """
        :param w2v_vocab: Dict mapping strings to their w2v index (of the w2v_weights matrix passed to the constructor
        of the neural network class).
        :param class_vocab: Dictionary that maps classes from column "concepts" to an integer, their index.
        :param c2v_vocab: Dict mapping chars to their c2v index (of the c2v_weights matrix passed to the constructor
        of the neural network class).
        :param sentence_length_cap: Sentences shorter than this cap will be padded, if longer will be cut.
        :param word_length_cap: Words shorter than this cap will be padded, if longer will be cut, only used if
        c2v embeddings are being used.
        :param add_matrix: If True, the transformed sample will also have a key "sequence_extra" which maps
        to the sentence seen as a matrix of shape (padded number of words, embedding size).
        """
        self.w2v_vocab = w2v_vocab
        self.c2v_vocab = c2v_vocab
        self.class_vocab = class_vocab
        self.pad_sentence_length = sentence_length_cap
        self.pad_word_length = word_length_cap
        self.add_matrix = add_matrix

    def _to_w2v_indexes(self, sentence):
        """
        Given a list of strings returns a tensor of shape (length of sentence).
        For each string in the sentence the corresponding w2v index that is going to be part of the final tensor
        is obtained from the w2v vocabulary if there exist a word-index pair, otherwise the word is either treated
        as <unk>, returning the <unk> idx.
        :param sentence: List of strings.
        :return: Tensor of shape (length of sentence) containing w2v indexes for each word in the sentence.
        """

        vectors = []
        for word in sentence:
            if word in self.w2v_vocab:
                vectors.append(self.w2v_vocab[word])
            elif word.title() in self.w2v_vocab:
                vectors.append(self.w2v_vocab[word.title()])
            elif word.isdigit() or word.find("DIGIT") != -1:
                vectors.append(self.w2v_vocab["number"])
            else:
                vectors.append(self.w2v_vocab["<UNK>"])

        if len(vectors) > self.pad_sentence_length:
            vectors = vectors[:self.pad_sentence_length]
        elif len(vectors) < self.pad_sentence_length:
            vectors.extend([self.w2v_vocab["<padding>"]] * (self.pad_sentence_length - len(vectors)))
        return torch.LongTensor(vectors)

    def _to_vocab_indexes(self, dictionary, sentence):
        """
        Given a dictionary mapping words to an index returns a tensor of shape (length of sentence) containing
        the indexes of those words.
        Assumes each word in the sentence has a mapping (this is usually used for concepts).
        :param dictionary: Dict mapping words in sentence to an index, must contain every word in sentence.
        :param sentence: List of strings.
        :return: Tensor of indexes.
        """
        idxs = [dictionary[word] for word in sentence]
        if len(idxs) > self.pad_sentence_length:
            idxs = idxs[:self.pad_sentence_length]
        elif len(idxs) < self.pad_sentence_length:
            idxs.extend([-1] * (self.pad_sentence_length - len(idxs)))
        return torch.tensor(idxs, dtype=torch.long)

    def _to_matrix(self, sentence, vocab, pad_length):
        """
        Given a list of strings returns a tensor of shape (1, padded length, 1).
        The length of the sentence is padded if it does not reach pad_length.
        For each string in the sentence the corresponding w2v (or c2v) index that is going to be part of the final tensor
        is obtained from the vocabulary if there exist a word-index pair, otherwise the word is either treated
        as <UNK> or <padding> (if it is a padding word) and their index is used.
        :param sentence: List of strings.
        :param vocab: Dict mapping strings to an index.
        :param pad_length: Length to which sentences (list of strings) will be padded to by using the index vocab["<padding>"].
        :return: Tensor of shape (1, pad_length, 1) containing w2v indexes for each word in the sentence.
        """

        vectors = []
        for word in sentence:
            if word in vocab:
                vectors.append(vocab[word])
            elif word.title() in vocab:
                vectors.append(vocab[word.title()])
            elif word.isdigit() or word.find("DIGIT") != -1:  # or any(char.isdigit() for char in word):
                vectors.append(self.w2v_vocab["number"])
            else:
                vectors.append(vocab["<UNK>"])
        if len(vectors) > pad_length:
            vectors = vectors[:pad_length]
        elif len(vectors) < pad_length:
            vectors.extend([vocab["<padding>"]] * (pad_length - len(vectors)))
        tensor = torch.tensor(vectors).view(1, pad_length)
        return tensor

    def _words_to_char_embeddings(self, sentence):
        """
        Given a sentence, return a tensor of size (1, 1, padded sentence length, padded word length) where
        values are indexes of char embeddings from the c2v vocab.
        :param sentence: List of strings.
        :return:
        """
        tensors_list = []
        res = torch.zeros(1, 1, self.pad_sentence_length, self.pad_word_length).long()
        curr_word = 0
        for word in sentence:
            char_matrix = self._to_matrix(word, self.c2v_vocab, self.pad_word_length)
            res[0, 0, curr_word, :] = char_matrix[0, :]
            tensors_list.append(char_matrix)
            curr_word += 1

        return res

    def __call__(self, sample):
        """
        Given a sample, a dict which has keys:
        "tokens" : mapping to list of strings (tokens)
        "concepts" : mapping to list of strings (concepts)
        return a transformed sample, which is another dict, which has keys:
        "tokens" : mapping to a list of w2v indices of the tokens in the sample
        "concepts" : mapping to a list of indices which represent classes (concepts in the sample)
        if the instance of class was init with add_matrix=True the transformed sample will also contain:
            "sequence_extra": mapping to a matrix of shape (1, length of words (padded)), containing w2v indices of the tokens
        if the instance of class was init with c2v_vocab different than None the transformed sample will also contain:
            "chars" : mapping to a matrix of #tokens in the sentence (padded) * characters in each token (padded), in order
            to later use c2v embeddings with convolution 
        passed with the sample, put in this way in order to later use w2v embeddings with convolution.
        :param sample:dict with sequence and concepts keys, mapping to list of strings.
        :return:
        """
        tsample = dict()
        tsample["tokens"] = self._to_w2v_indexes(sample["tokens"])
        tsample["concepts"] = self._to_vocab_indexes(self.class_vocab, sample["concepts"])
        if self.add_matrix:
            tsample["sequence_extra"] = self._to_matrix(sample["tokens"], self.w2v_vocab, self.pad_sentence_length)
        if self.c2v_vocab is not None:
            tsample["chars"] = self._words_to_char_embeddings(sample["tokens"])
        return tsample
