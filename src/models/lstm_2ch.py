import torch
import torch.nn as nn
import torch.nn.functional as F

import data_manager


class LSTMN2CH(nn.Module):
    def __init__(self, device, w2v_weights, hidden_dim, tagset_size, drop_rate, bidirectional=False,
                 embedding_norm=10.):
        """
        :param device: Device to which to map tensors (GPU or CPU).
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param hidden_dim Size of the hidden dimension of the recurrent layer.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the recurrent should be bidirectional.
        :param embedding_norm: Max norm of the dynamic embeddings.
        """
        super(LSTMN2CH, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.embedding_dim = w2v_weights.shape[1]
        self.w2v_weights = w2v_weights
        self.bidirectional = bidirectional

        self.embedding_static = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=True)
        self.embedding_dyn = nn.Embedding(w2v_weights.shape[0], w2v_weights.shape[1], max_norm=embedding_norm,
                                          scale_grad_by_freq=True)

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)

        # two "parallel" recurrent layers, 1 for static and 1 for dynamic embeddings
        self.recurrent_static = nn.LSTM(self.embedding_dim, self.hidden_dim // (2 if not bidirectional else 4),
                                        batch_first=True, bidirectional=bidirectional)
        self.recurrent_dyn = nn.LSTM(self.embedding_dim, self.hidden_dim // (2 if not bidirectional else 4),
                                     batch_first=True, bidirectional=bidirectional)

        self.hidden2tag = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.ReLU(inplace=True)
        )

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the recurrent layer.
        :param batch_size
        :return: Initialized hidden state of the recurrent encoder.
        """
        if self.bidirectional:
            state = [
                torch.zeros(self.recurrent_static.num_layers * 2, batch_size, self.hidden_dim // 4).to(self.device),
                torch.zeros(self.recurrent_static.num_layers * 2, batch_size, self.hidden_dim // 4).to(self.device)]
        else:
            state = [torch.zeros(self.recurrent_static.num_layers, batch_size, self.hidden_dim // 2).to(self.device),
                     torch.zeros(self.recurrent_static.num_layers, batch_size, self.hidden_dim // 2).to(self.device)]
        return state

    def forward(self, batch):
        hidden_static = self.init_hidden(len(batch))
        hidden_dyn = self.init_hidden(len(batch))

        # embed using static embeddings and pass through the recurrent layer
        data, labels, char_data = data_manager.batch_sequence(batch, self.device)
        data_static = self.embedding_static(data)
        data_static = self.drop(data_static)
        lstm_out_static, hidden_static = self.recurrent_static(data_static, hidden_static)

        # embed using dynamic embeddings and pass through the recurrent layer
        data_dynamic = self.embedding_dyn(data)
        data_dynamic = self.drop(data_dynamic)
        lstm_out_dyn, hidden_dyn = self.recurrent_dyn(data_dynamic, hidden_dyn)

        # concatenate results
        output = torch.cat([lstm_out_static, lstm_out_dyn], dim=2)

        # send output to fc layer(s)
        tag_space = self.hidden2tag(output.unsqueeze(1).contiguous())
        tag_scores = F.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
