import torch
import torch.nn as nn
import torch.nn.functional as F

import data_manager


class GRU(nn.Module):
    def __init__(self, w2v_weights, gru_layers, hidden_dim, tagset_size, drop_rate, bidirectional=False,
                 freeze=True, embedding_norm=10.):
        """
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param gru_layers: Number of gru layers.
        :param hidden_dim Size of the hidden dimension of the gru layer.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the gru layer should be bidirectional.
        :param freeze: If the embedding parameters should be frozen or trained during training.
        :param embedding_norm: Max norm of the embeddings.
        """
        super(GRU, self).__init__()

        self.w2v_weights = w2v_weights
        self.embedding_dim = w2v_weights.shape[1]
        self.gru_layers = gru_layers
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        self.drop_rate = drop_rate

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights).cuda(), freeze=freeze)
        self.embedding.max_norm = embedding_norm

        self.drop = nn.Dropout(self.drop_rate)

        # The gru takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim // (1 if not bidirectional else 2),
                          dropout=self.drop_rate, batch_first=True, num_layers=self.gru_layers,
                          bidirectional=bidirectional)
        if bidirectional:
            self.init_hidden = self.__init_hidden_bidirectional

        self.hidden2tag = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.ReLU(inplace=True)
        )

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the gru layer.
        :param batch_size
        :return: Initialized hidden state of the gru.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

    def __init_hidden_bidirectional(self, batch_size):
        """
        Inits the hidden state of the gru layer.
        :param batch_size
        :return: Initialized hidden state of the gru.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(self.gru.num_layers * 2, batch_size, self.hidden_dim // 2).cuda()

    def forward(self, batch):
        hidden = self.init_hidden(len(batch))

        # pack sentences and pass through rnn
        data, labels, char_data = data_manager.batch_sequence(batch)
        data = self.embedding(data)
        data = self.drop(data)

        gru_out, hidden = self.gru(data, hidden)

        # send output to fc layer(s)
        tag_space = self.hidden2tag(gru_out.unsqueeze(1).contiguous())
        tag_scores = F.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
