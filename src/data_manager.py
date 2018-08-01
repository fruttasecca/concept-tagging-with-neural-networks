import random
import pandas as pd
import numpy as np
import collections  # to make the class w2v_vocab read only
import torch
from torch.utils.data import Dataset


class PytorchDataset(Dataset):
    """Dataset to import augmented data."""

    def __init__(self, pickle, init_transform, getitem_transform=None):
        """
        :param pickle Path to a pickle or a list of paths to pickles containing a dataframe with tokens and
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

        self.class_vocab = dict()
        for _, data_point in df.iterrows():
            for concept in data_point["concepts"]:
                if concept not in self.class_vocab:
                    self.class_vocab[concept] = len(self.class_vocab)

        # transform and save data
        self.data = dict()
        for i in range(len(df)):
            sample = df.iloc[i, :]
            self.data[i] = self.init_transform(sample)

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
                        'I-movie.gross_revenue': 25, 'I-country.name': 22, 'B-person.nationality': 39,
                        'I-director.name': 16,
                        'I-award.category': 42, 'I-movie.subject': 5, 'B-movie.type': 41}

class_vocab_movies = DictWrapper(__class_vocab_movies)

__class_vocab_atis = {'I-airline_name': 18, 'B-stoploc.airport_code': 123, 'B-state_code': 88, 'B-meal_description': 76,
                      'B-fromloc.city_name': 3, 'B-day_number': 118, 'B-depart_date.month_name': 25,
                      'I-return_date.date_relative': 116,
                      'B-meal': 66, 'I-arrive_date.day_number': 34, 'B-depart_date.year': 62,
                      'B-fromloc.state_name': 41, 'B-mod': 77,
                      'B-depart_date.today_relative': 46, 'B-arrive_time.time_relative': 22,
                      'B-return_date.today_relative': 112,
                      'I-fromloc.city_name': 4, 'B-flight_mod': 47, 'B-day_name': 100, 'B-flight': 126,
                      'B-arrive_time.end_time': 55,
                      'I-flight_time': 84, 'B-flight_number': 2, 'I-fare_basis_code': 98,
                      'B-depart_time.start_time': 81,
                      'I-state_name': 124, 'B-arrive_date.day_name': 10, 'B-aircraft_code': 72, 'I-transport_type': 74,
                      'I-cost_relative': 15, 'B-arrive_time.start_time': 53, 'B-arrive_time.period_of_day': 50,
                      'I-time': 120, 'B-depart_date.date_relative': 51, 'B-flight_days': 59,
                      'I-return_date.day_number': 95,
                      'B-return_date.month_name': 93, 'B-return_date.day_name': 97, 'B-flight_stop': 21,
                      'I-toloc.city_name': 6,
                      'B-depart_time.time': 38, 'B-toloc.airport_name': 68, 'I-airport_name': 64, 'B-or': 44,
                      'B-arrive_time.period_mod': 9,
                      'I-arrive_time.time_relative': 89, 'B-airline_name': 1, 'B-depart_time.period_of_day': 11,
                      'B-economy': 60,
                      'B-compartment': 122, 'B-fromloc.airport_name': 12, 'I-fare_amount': 36, 'B-state_name': 105,
                      'B-stoploc.state_code': 20, 'B-class_type': 24, 'B-round_trip': 16, 'I-round_trip': 17,
                      'B-depart_time.period_mod': 40,
                      'I-depart_date.today_relative': 102, 'I-flight_stop': 75, 'I-return_date.today_relative': 113,
                      'I-class_type': 45, 'B-time': 110, 'B-stoploc.city_name': 19, 'B-fromloc.airport_code': 52,
                      'I-arrive_time.period_of_day': 108, 'I-depart_time.start_time': 86, 'B-restriction_code': 14,
                      'B-toloc.state_code': 29, 'B-fare_basis_code': 58, 'B-airport_code': 78,
                      'B-fromloc.state_code': 57,
                      'I-city_name': 31, 'B-fare_amount': 35, 'B-today_relative': 106, 'B-meal_code': 103,
                      'I-fromloc.airport_name': 13,
                      'B-days_code': 109, 'I-arrive_time.time': 28, 'I-flight_mod': 67, 'I-flight_number': 121,
                      'B-airport_name': 63, 'B-depart_date.day_number': 26, 'I-toloc.state_name': 80,
                      'B-transport_type': 73,
                      'B-arrive_date.today_relative': 85, 'I-stoploc.city_name': 65, 'I-meal_description': 114,
                      'B-arrive_time.time': 23,
                      'I-toloc.airport_name': 69, 'B-return_date.day_number': 94, 'B-return_time.period_of_day': 92,
                      'B-period_of_day': 101,
                      'B-cost_relative': 8, 'I-depart_time.time': 39, 'I-economy': 71, 'I-meal_code': 104,
                      'B-depart_date.day_name': 7,
                      'B-stoploc.airport_name': 111, 'I-today_relative': 107, 'I-depart_time.period_of_day': 99,
                      'B-toloc.city_name': 5, 'B-depart_time.end_time': 82, 'O': 0, 'B-arrive_date.month_name': 32,
                      'B-city_name': 27, 'B-time_relative': 119, 'B-arrive_date.date_relative': 90,
                      'I-arrive_time.end_time': 56,
                      'I-restriction_code': 70, 'B-depart_time.time_relative': 37, 'B-connect': 49,
                      'B-return_time.period_mod': 91,
                      'B-toloc.country_name': 96, 'B-booking_class': 125, 'B-month_name': 117, 'B-flight_time': 48,
                      'B-return_date.date_relative': 79, 'B-toloc.state_name': 43, 'I-depart_time.time_relative': 115,
                      'I-depart_date.day_number': 61, 'B-airline_code': 30, 'B-arrive_date.day_number': 33,
                      'I-fromloc.state_name': 42, 'I-arrive_time.start_time': 54, 'B-toloc.airport_code': 83,
                      'I-depart_time.end_time': 87}

class_vocab_atis = DictWrapper(__class_vocab_atis)


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
    """ Transformer class to be passed to the pytorch dataset class to transform data at import time. """

    def __init__(self, device, w2v_vocab, class_vocab, c2v_vocab=None, sentence_length_cap=50, word_length_cap=30, add_matrix=True):
        """
        :param w2v_vocab: Dict mapping strings to their w2v index (of the w2v_weights matrix passed to the constructor
        of the neural network class).
        :param class_vocab: Dictionary that maps classes from column "concepts" to an integer.
        :param c2v_vocab: Dict mapping chars to their c2v index (of the c2v_weights matrix passed to the constructor
        of the neural network class).
        """
        self.device = device
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
            elif word == "@card@":
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
