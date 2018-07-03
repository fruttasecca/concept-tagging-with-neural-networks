import random

import pandas as pd
import numpy as np
import collections  # to make the class w2v_vocab read only
import torch
from torch.utils.data import Dataset


class pytorch_dataset(Dataset):
    """Dataset to import augmented data."""

    def __init__(self, pickle, init_transform, getitem_transform=None):
        """
        :param pickle Path to a pickle or a list of paths to pickles containing a dataframe with tokens, lemmas, pos,
        concepts columns containing the data.
        :param init_transform: Transform function to be used on data points at import time.
        :param getitem_transform: Transform function to be used on data points when they are retrieved with __getitem__.
        """
        print("loading data:\n %s" % pickle)
        if type(pickle) == type([]):
            df_list = []
            for name in pickle:
                df_list.append(pd.read_pickle(name))
            df = pd.concat(df_list)
        else:
            df = pd.read_pickle(pickle)

        self.init_transform = init_transform
        self.getitem_transform = getitem_transform

        self.pos_vocab = dict()
        self.class_vocab = dict()
        for _, data_point in df.iterrows():
            for concept in data_point["concepts"]:
                if concept not in self.class_vocab:
                    self.class_vocab[concept] = len(self.class_vocab)
            for pos in data_point["pos"]:
                if pos not in self.pos_vocab:
                    self.pos_vocab[pos] = len(self.pos_vocab)

        # transform and save data
        self.data = dict()
        for i in range(len(df)):
            sample = df.iloc[i, :]
            self.data[i] = self.init_transform(sample, self.pos_vocab)

        # empty dataframe
        df = df.iloc[0:0]

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
    w2v["<unk>"] = len(w2v_weights) - 2
    w2v["<padding>"] = len(w2v_weights) - 1
    return w2v, w2v_weights


class Data(object):
    """
    Class to contain data, once initialized it is organized in this way:
    - properties that allow you to get counter maps for words, lemmas, pos tags, concepts, and pairs
    - properties that allow you to get words, lemmas, pos tags, concepts lexicon
    - methods that allow you to get the counter for a specific word, lemma, pos tags, concept or pair
    note: this class is something i copy pasted from an old project of mine, not really that useful in here
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
        self.__lemmas_counter = dict()
        self.__pos_counter = dict()
        self.__concepts_counter = dict()
        self.__concepts_clean_counter = dict()  # concepts without IOB notation
        # pairs of stuff
        self.__word_concept_counter = dict()
        self.__lemma_concept_counter = dict()
        self.__pos_concept_counter = dict()

        for phrase in self._data:
            for data_point in phrase:
                word, lemma, pos, concept = data_point
                # singletons
                self.__words_counter[word] = 1 + self.__words_counter.get(word, 0)
                self.__lemmas_counter[lemma] = 1 + self.__lemmas_counter.get(lemma, 0)
                self.__pos_counter[pos] = 1 + self.__pos_counter.get(pos, 0)
                self.__concepts_counter[concept] = 1 + self.__concepts_counter.get(concept, 0)
                clean_c = concept if concept == "O" else concept[2:]
                self.__concepts_clean_counter[clean_c] = 1 + self.__concepts_clean_counter.get(clean_c, 0)
                # pairs of stuff
                self.__word_concept_counter[word + " " + concept] = 1 + self.__word_concept_counter.get(
                    word + " " + concept, 0)
                self.__lemma_concept_counter[lemma + " " + concept] = 1 + self.__lemma_concept_counter.get(
                    lemma + " " + concept, 0)
                self.__pos_concept_counter[pos + " " + concept] = 1 + self.__pos_concept_counter.get(
                    pos + " " + concept, 0)

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
    def counter_lemmas(self):
        """
        :return: Dictionary that maps a lemma to its counter.
        """
        return self.__lemmas_counter

    @property
    def counter_pos(self):
        """
        :return: Dictionary that maps a pos tag to its counter.
        """
        return self.__pos_counter

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
    def counter_lemma_concept(self):
        """
        :return: Dictionary that maps a lemma + concept pair to its counter, separated by space.
        """
        return self.__lemma_concept_counter

    @property
    def counter_pos_concept(self):
        """
        :return: Dictionary that maps a pos tag + concept pair to its counter, separated by space.
        """
        return self.__pos_concept_counter

    @property
    def lexicon_words(self):
        """
        :return: List of words in the corpus.<epsilon> and <unk> not included.
        """
        return list(self.counter_words.keys())

    @property
    def lexicon_lemmas(self):
        """
        :return: List of lemmas in the corpus.<epsilon> and <unk> not included.
        """
        return list(self.counter_lemmas.keys())

    @property
    def lexicon_pos(self):
        """
        :return: List of pos tags in the corpus.<epsilon> and <unk> not included.
        """
        return list(self.counter_pos.keys())

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

    def lemma(self, lemma):
        """
        :param lemma: Lemma for which to return the count for.
        :return: Count of the lemma, >= 0.
        """
        return self.__lemmas_counter.get(lemma, 0)

    def pos(self, pos):
        """
        :param pos: Pos tag for which to return the count for.
        :return: Count of the pos tag, >= 0.
        """
        return self.__pos_counter.get(pos, 0)

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

    def lemma_concept(self, lemma, concept):
        """
        :param lemma: Lemma of the lemma - concept pair.
        :param concept: Concept of the lemma - concept pair.
        :return: Count of the lemma - concept pair >= 0.
        """
        return self.__lemma_concept_counter.get(lemma + " " + concept, 0)

    def pos_concept(self, pos, concept):
        """
        :param pos: Pos tag of the pos - concept pair.
        :param concept: Concept of the pos - concept pair.
        :return: Count of the pos - concept pair >= 0.
        """
        return self.__pos_concept_counter.get(pos + " " + concept, 0)

    def to_dataframe(self):
        """
        Transform the data to a df containing the tokens, lemmas, pos and concepts columns.
        Each sentence will correspond to a row, each column (for each row) contains a list of strings.
        :return: Dataframe transposition of this data object.
        """
        df = pd.DataFrame(columns=["tokens", "lemmas", "pos", "concepts"])
        for i, phrase in enumerate(self._data):
            words = []
            lemmas = []
            pos_tags = []
            concepts = []
            for data_point in phrase:
                word, lemma, pos, concept = data_point
                words.append(word)
                lemmas.append(lemma)
                pos_tags.append(pos)
                concepts.append(concept)
            df.loc[i] = [words, lemmas, pos_tags, concepts]
        return df


# simple class to make our class vocabulary read only
class DictWrapper(collections.Mapping):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


__class_vocab_movies = {'I-movie.name': 2, 'I-character.name': 17, 'I-movie.location': 36, 'B-movie.location': 30,
                 'B-movie.name': 1,
                 'B-director.name': 15, 'B-person.name': 6, 'I-actor.name': 20, 'B-movie.star_rating': 37,
                 'B-actor.name': 19,
                 'B-award.ceremony': 27, 'I-rating.name': 28, 'B-director.nationality': 31,
                 'B-movie.release_region': 35,
                 'B-actor.nationality': 29, 'I-producer.name': 23, 'B-character.name': 9, 'B-producer.name': 11,
                 'B-movie.genre': 14, 'I-movie.release_date': 13, 'I-movie.language': 21, 'B-actor.type': 18,
                 'B-movie.description': 26, 'I-person.name': 7, 'I-movie.genre': 34, 'B-award.category': 33,
                 'B-movie.language': 10,
                 'O': 0, 'B-country.name': 3, 'B-rating.name': 8, 'I-award.ceremony': 32, 'B-movie.subject': 4,
                 'B-movie.release_date': 12, 'B-movie.gross_revenue': 24, 'I-movie.release_region': 38,
                 'I-actor.nationality': 40,
                 'I-movie.gross_revenue': 25, 'I-country.name': 22, 'B-person.nationality': 39, 'I-director.name': 16,
                 'I-award.category': 42, 'I-movie.subject': 5, 'B-movie.type': 41}
class_vocab_movies = DictWrapper(__class_vocab_movies)


def batch_sequence(batch):
    """
    Given a batch return sequence data, labels and chars data (if present) in a "batched" way, as a tensor
    where the first dimension is the dimension of the batch.
    :param batch: List of sample points, each sample point should contain "sequence" and "concepts" data, list
    of integers, it may also contain "chars" data, which is a list of integers.
    """
    list_of_data_tensors = [sample["sequence"].unsqueeze(0) for sample in batch]
    data = torch.cat(list_of_data_tensors, dim=0).cuda()
    list_of_labels_tensors = [sample["concepts"].unsqueeze(0) for sample in batch]
    labels = torch.cat(list_of_labels_tensors, dim=0).cuda()
    char_data = None
    if "chars" in batch[0]:
        list_of_char_data_tensors = [sample["chars"] for sample in batch]
        char_data = torch.cat(list_of_char_data_tensors, dim=0).cuda()
    return data, labels, char_data


class DropTransform(object):
    """ Transformer class to be passed to the pytorch dataset class to transform data at run time, it randomly
    drops word indexes to 'simulate' unknown words."""

    def __init__(self, drop_chance, unk_idx, ignore_idx):
        """
        :param drop_chance: Chance of dropping a word.
        :param unk_idx: Which index to use in place of the one of the dropped word.
        :param ignore_idx: Index to never drop (i.e. the padding index).
        """
        self.drop_chance = drop_chance
        self.unk_idx = unk_idx
        self.ignore_idx = ignore_idx

    def might_drop(self, idx):
        return self.unk_idx if (random.uniform(0, 1) < self.drop_chance and idx != self.ignore_idx) else idx

    def __call__(self, sample):
        tsample = dict()
        seq = sample["sequence"].clone()
        for i in range(len(seq)):
            seq[i] = self.might_drop(seq[i].item())
        tsample["sequence"] = seq
        tsample["pos"] = sample["pos"]
        tsample["concepts"] = sample["concepts"]
        tsample["sequence_extra"] = sample["sequence_extra"]
        if "chars" in sample:
            tsample["chars"] = sample["chars"]
        return tsample


corrected = {
    "pg-13": "rating",
    "r-rated": "rating",
    "nc-17": "rating",
    "g-rated": "rating",
    "paranorman": "paranormal",
    "seventieseven": "number",
    "adventeurous": "adventurous",
    "beautfiul": "beautiful",
    "translyvania": "transylvania",
    "descrbe": "describe",
    "realese": "release",
    "japaneese": "japanese",
    "spilberg": "spielberg",
    "terantino": "tarantino",
    "realase": "release",
    "procuce": "produced",
    "charactors": "characters",
    "scorscese": "scorsese",
    "transylavania": "transylvania",
    "highest-grossing": "grossing",
    "antogonist": "antagoinsit",
    "directort": "director",
    "funnie": "funny",
    "co-star": "star",
    "avergers": "avengers",
    "!": "exclamation_mark",
    ":": "punctuation",
    "-": "punctuation",
}


class InitTransform(object):
    """ Transformer class to be passed to the pytorch dataset class to transform data at import time. """

    def __init__(self, sequence, w2v_vocab, class_vocab, c2v_vocab=None):
        """
        :param sequence: Which sequence data to use.
        :param w2v_vocab: Dict mapping strings to their w2v index (of the w2v_weights matrix passed to the constructor
        of the neural network class).
        :param class_vocab: Dictionary that maps classes from column "concepts" to an integer.
        :param w2v_vocab: Dict mapping chars to their c2v index (of the c2v_weights matrix passed to the constructor
        of the neural network class).
        """
        self.w2v_vocab = w2v_vocab
        self.c2v_vocab = c2v_vocab
        self.sequence = sequence
        self.class_vocab = class_vocab
        self.pad_sentence_length = 25  # cap to longest sequence
        self.pad_word_length = 16
        global corrected
        self.corrected = corrected

    def to_w2v_indexes(self, sentence):
        """
        Given a list of strings returns a tensor of shape (length of sentence).
        For each string in the sentence the corresponding w2v index that is going to be part of the final tensor
        is obtained from the w2v vocabulary if there exist a word-index pair, otherwise the word is either treated
        as <unk>.
        :param sentence: List of strings.
        :return: Tensor of shape (length of sentence) containing w2v indexes for each word in the sentence.
        """

        vectors = []
        for word in sentence:
            if word in self.corrected:
                word = self.corrected[word]
            if word in self.w2v_vocab:
                vectors.append(self.w2v_vocab[word])
            elif word.title() in self.w2v_vocab:
                vectors.append(self.w2v_vocab[word.title()])
            elif word.isdigit():
                vectors.append(self.w2v_vocab["number"])
            elif word == "@card@":
                vectors.append(self.w2v_vocab["number"])
            else:
                vectors.append(self.w2v_vocab["<unk>"])

        if len(vectors) > self.pad_sentence_length:
            vectors = vectors[:self.pad_sentence_length]
        elif len(vectors) < self.pad_sentence_length:
            vectors.extend([self.w2v_vocab["<padding>"]] * (self.pad_sentence_length - len(vectors)))
        return torch.LongTensor(vectors)

    def to_vocab_indexes(self, dictionary, sentence):
        """
        Given a dictionary mapping words to an index returns a tensor of shape (length of sentence) containing
        the indexes of those words.
        Assumes each word in the sentence has a mapping.
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

    def to_matrix(self, sentence, vocab, pad_length):
        """
        Given a list of strings returns a tensor of shape (1, padded length, 1).
        The length of the sentence is padded if it does not reach pad_length.
        For each string in the sentence the corresponding w2v (or c2v) index that is going to be part of the final tensor
        is obtained from the ocabulary if there exist a word-index pair, otherwise the word is either treated
        as <unk> or <padding> (if it is a padding word) and their index is used.
        :param sentence: List of strings.
        :param vocab: Dict mapping strings to an index.
        :param pad_length: Length to which sentences (list of strings) will be padded to by using the index vocab["<padding>"].
        :return: Tensor of shape (1, pad_length, 1) containing w2v indexes for each word in the sentence.
        """

        vectors = []
        for word in sentence:
            if word in self.corrected:
                word = self.corrected[word]
            if word in vocab:
                vectors.append(vocab[word])
            elif word.title() in vocab:
                vectors.append(vocab[word.title()])
            elif word.isdigit():
                vectors.append(self.w2v_vocab["number"])
            elif word == "@card@":
                vectors.append(self.w2v_vocab["number"])
            else:
                vectors.append(vocab["<unk>"])
        if len(vectors) > pad_length:
            vectors = vectors[:pad_length]
        elif len(vectors) < pad_length:
            vectors.extend([vocab["<padding>"]] * (pad_length - len(vectors)))
        tensor = torch.tensor(vectors).view(1, pad_length)
        return tensor

    def words_to_char_embeddings(self, sentence):
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
            char_matrix = self.to_matrix(word, self.c2v_vocab, self.pad_word_length)
            res[0, 0, curr_word, :] = char_matrix[0, :]
            tensors_list.append(char_matrix)
            curr_word += 1

        return res

    def __call__(self, sample, pos_vocab):
        tsample = dict()
        tsample["sequence"] = self.to_w2v_indexes(sample[self.sequence])
        tsample["pos"] = self.to_vocab_indexes(pos_vocab, sample["pos"])
        tsample["concepts"] = self.to_vocab_indexes(self.class_vocab, sample["concepts"])
        tsample["sequence_extra"] = self.to_matrix(sample[self.sequence], self.w2v_vocab, self.pad_sentence_length)
        if self.c2v_vocab is not None:
            tsample["chars"] = self.words_to_char_embeddings(sample[self.sequence])
        return tsample
