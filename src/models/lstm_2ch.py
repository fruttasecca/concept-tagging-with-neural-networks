import torch
import torch.nn as nn
import torch.nn.functional as F

import data_manager


class LSTMN2CH(nn.Module):
    def __init__(self, w2v_weights, lstm_layers, hidden_dim, tagset_size, drop_rate, bidirectional=False, embedding_norm=10.):
        """
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param lstm_layers: Number of lstm layers.
        :param hidden_dim Size of the hidden dimension of the lstm layer.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the lstm should be bidirectional.
        :param embedding_norm: Max norm of the dynamic embeddings.
        """
        super(LSTMN2CH, self).__init__()

        self.w2v_weights = w2v_weights
        self.embedding_dim = w2v_weights.shape[1]
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        self.drop_rate = drop_rate

        self.embedding_static = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights).cuda(), freeze=True)
        self.embedding_dyn = nn.Embedding(w2v_weights.shape[0], w2v_weights.shape[1], max_norm=embedding_norm,
                                          scale_grad_by_freq=True)

        self.drop = nn.Dropout(self.drop_rate)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_static = nn.LSTM(self.embedding_dim, self.hidden_dim // (2 if not bidirectional else 4),
                                   dropout=self.drop_rate, batch_first=True,
                                   num_layers=self.lstm_layers, bidirectional=bidirectional)
        self.lstm_dyn = nn.LSTM(self.embedding_dim, self.hidden_dim // (2 if not bidirectional else 4),
                                dropout=self.drop_rate, batch_first=True,
                                num_layers=self.lstm_layers, bidirectional=bidirectional)
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
        Inits the hidden state of the lstm layers, both static and dynamic.
        :param batch_size
        :return: A pair containing the intialized hidden states both static and dynamic.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers, batch_size, self.hidden_dim // 2).cuda(),
                torch.zeros(self.lstm_layers, batch_size, self.hidden_dim // 2).cuda()), (
                   torch.zeros(self.lstm_layers, batch_size, self.hidden_dim // 2).cuda(),
                   torch.zeros(self.lstm_layers, batch_size, self.hidden_dim // 2).cuda())

    def __init_hidden_bidirectional(self, batch_size):
        """
        Inits the hidden state of the lstm layers, both static and dynamic.
        :param batch_size
        :return: A pair containing the intialized hidden states both static and dynamic.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim // 4).cuda(),
                torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim // 4).cuda()), (
                   torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim // 4).cuda(),
                   torch.zeros(self.lstm_layers * 2, batch_size, self.hidden_dim // 4).cuda())

    def forward(self, batch):
        hidden_static, hidden_dyn = self.init_hidden(len(batch))

        # embed on static and pass through lstm
        # pack sentences and pass through rnn
        data, labels, char_data = data_manager.batch_sequence(batch)
        data_static = self.embedding_static(data)
        data_static = self.drop(data_static)
        lstm_out_static, hidden_static = self.lstm_static(data_static, hidden_static)

        # embed on dynamic and pass through lstm
        data_dynamic = self.embedding_dyn(data)
        data_dynamic = self.drop(data_dynamic)
        lstm_out_dyn, hidden_dyn = self.lstm_dyn(data_dynamic, hidden_dyn)

        output = torch.cat([lstm_out_static, lstm_out_dyn], dim=2)
        # send output to fc layer(s)
        tag_space = self.hidden2tag(output.unsqueeze(1).contiguous())
        tag_scores = F.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
