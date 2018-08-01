import torch
import torch.nn as nn
import torch.nn.functional as F

import data_manager


class LSTM(nn.Module):
    def __init__(self, device, w2v_weights, hidden_dim, tagset_size, drop_rate, bidirectional=False,
                 freeze=True, embedding_norm=10., c2v_weights=None, pad_word_length=16):
        """
        :param device: Device to which to map tensors (GPU or CPU).
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param hidden_dim Size of the hidden dimension of the recurrent layer.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the recurrent should be bidirectional.
        :param freeze: If the embedding parameters should be frozen or trained during training.
        :param embedding_norm: Max norm of the embeddings.
        :param c2v_weights: Matrix of w2v c2v_weights, ith row contains the embedding for the char mapped to the ith index, the
        last row should correspond to the padding character, by passing this the nn will use a convolutional network
        on character representations add that the obtained feature vector to the embedding vector of the token.
        :param pad_word_length: Length to which each word is padded to, only used if c2v_weights has been passed and
        the network is going to use char representations, it is needed for the size of the maxpooling window.
        """
        super(LSTM, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.embedding_dim = w2v_weights.shape[1]
        self.w2v_weights = w2v_weights
        self.c2v_weights = c2v_weights
        self.bidirectional = bidirectional
        self.pad_word_length = pad_word_length
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
        self.embedding.max_norm = embedding_norm

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)

        # The recurrent layer takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.recurrent = nn.LSTM(self.embedding_dim, self.hidden_dim // (1 if not self.bidirectional else 2),
                                 batch_first=True, bidirectional=self.bidirectional)

        self.hidden2tag = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.ReLU(inplace=True)
        )

        # setup convolution on characters if c2v_weights are passed
        if self.c2v_weights is not None:
            self.char_embedding_dim = c2v_weights.shape[1]
            self.char_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(c2v_weights), freeze=freeze)
            self.char_embedding.max_norm = embedding_norm
            self.feats = 20  # for the output channels of the conv layers

            self.recurrent = nn.LSTM(self.embedding_dim + 50,
                                     self.hidden_dim // (1 if not self.bidirectional else 2),
                                     batch_first=True, bidirectional=self.bidirectional)

            # conv layers for single character, pairs of characters, 3x characters
            self.ngram1 = nn.Sequential(
                nn.Conv2d(1, self.feats * 1, kernel_size=(1, self.char_embedding_dim),
                          stride=(1, self.char_embedding_dim),
                          padding=0),
                nn.Dropout2d(p=self.drop_rate),
                nn.MaxPool2d(kernel_size=(self.pad_word_length, 1)),
                nn.Tanh(),
            )

            self.ngram2 = nn.Sequential(
                nn.Conv2d(1, self.feats * 2, kernel_size=(2, self.char_embedding_dim),
                          stride=(1, self.char_embedding_dim),
                          padding=0),
                nn.Dropout2d(p=self.drop_rate),
                nn.MaxPool2d(kernel_size=(self.pad_word_length - 1, 1)),
                nn.Tanh(),
            )

            self.ngram3 = nn.Sequential(
                nn.Conv2d(1, self.feats * 3, kernel_size=(3, self.char_embedding_dim),
                          stride=(1, self.char_embedding_dim),
                          padding=0),
                nn.Dropout2d(p=self.drop_rate),
                nn.MaxPool2d(kernel_size=(self.pad_word_length - 2, 1)),
                nn.Tanh(),
            )

            # seq layers to elaborate on the output of conv layers
            self.fc1 = nn.Sequential(
                nn.Linear(self.feats, 10),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(self.feats * 2, 20),
            )
            self.fc3 = nn.Sequential(
                nn.Linear(self.feats * 3, 20),
            )

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the recurrent layer.
        :param batch_size
        :return: Initialized hidden state of the recurrent encoder.
        """
        if self.bidirectional:
            state = [torch.zeros(self.recurrent.num_layers * 2, batch_size, self.hidden_dim // 2).to(self.device),
                     torch.zeros(self.recurrent.num_layers * 2, batch_size, self.hidden_dim // 2).to(self.device)]
        else:
            state = [torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_dim).to(self.device),
                     torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_dim).to(self.device)]
        return state

    def forward(self, batch):
        """
        Forward pass given data.
        :param batch: List of samples containing data as transformed by the init transformer of this class.
        :return: A (batch of) vectors of length equal to tagset, scoring each possible class for each word in a sentence,
        for all sentences; a tensor containing the true label for each word and a tensor containing the lengths
        of the sequences in descending order.
        """
        hidden = self.init_hidden(len(batch))

        # pack sentences and pass through rnn
        data, labels, char_data = data_manager.batch_sequence(batch, self.device)
        data = self.embedding(data)
        data = self.drop(data)

        if self.c2v_weights is not None:
            batched_conv = []
            char_data = self.char_embedding(char_data)
            char_data = self.drop(char_data)
            num_words = char_data.size()[2]
            for i in range(num_words):
                # get word for each batch, then convolute on the ith word of each batch and concatenate
                c = char_data[:, 0, i, :, :].unsqueeze(1)
                ngram1 = self.ngram1(c).view(char_data.size()[0], 1, 1, -1)
                ngram2 = self.ngram2(c).view(char_data.size()[0], 1, 1, -1)
                ngram3 = self.ngram3(c).view(char_data.size()[0], 1, 1, -1)
                ngram1 = self.fc1(ngram1)
                ngram2 = self.fc2(ngram2)
                ngram3 = self.fc3(ngram3)
                batched_conv.append(torch.cat([ngram1, ngram2, ngram3], dim=3))
            batched_conv = torch.cat(batched_conv, dim=1).squeeze(2)
            data = torch.cat([data, batched_conv], dim=2)

        rec_out, hidden = self.recurrent(data, hidden)
        # send output to fc layer(s)
        tag_space = self.hidden2tag(rec_out.unsqueeze(1).contiguous())
        tag_scores = F.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
