import torch
import torch.nn as nn

import data_manager


class INIT(nn.Module):
    def __init__(self, w2v_weights, hidden_dim, tagset_size, pad_sentence_length, drop_rate, bidirectional=False,
                 freeze=True, embedding_norm=10):
        """
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param hidden_dim The hidden memory of the recurrent layer will have a size of 3 times this.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the lstm should be bidirectional.
        :param freeze: If the embedding parameters should be frozen or trained during training.
        :param embedding_norm: Max norm of the embeddings.
        """
        super(INIT, self).__init__()

        self.pad_sentence_length = pad_sentence_length
        self.tagset_size = tagset_size
        self.w2v_weights = w2v_weights

        self.bidirectional = bidirectional
        self.embedding_dim = w2v_weights.shape[1]

        self.hidden_dim = hidden_dim
        self.fc_drop = drop_rate

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
        self.embedding.max_norm = embedding_norm

        self.drop = nn.Dropout(self.fc_drop)

        fc_feats = 200
        self.fc = nn.Sequential(
            nn.Dropout(self.fc_drop),
            nn.Linear(self.pad_sentence_length * self.embedding_dim, fc_feats),
            nn.BatchNorm1d(fc_feats),
            nn.ReLU(),
            nn.Dropout(self.fc_drop),
            nn.Linear(fc_feats, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
        )

        self.to_tag_space = nn.Sequential(
            nn.Dropout(self.fc_drop),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.GRU(self.embedding_dim, self.hidden_dim // (1 if not bidirectional else 2),
                           dropout=drop_rate, batch_first=True, bidirectional=bidirectional)

    @staticmethod
    def prepare_batch(batch):
        seq_list = []
        for sample in batch:
            seq = sample["sequence_extra"].unsqueeze(1).cuda()
            seq_list.append(seq)
        res_seq = torch.cat(seq_list, dim=0)
        return res_seq

    def forward(self, batch):
        # pre-elaborate hidden state
        sequence = self.prepare_batch(batch)
        embedded = self.embedding(sequence).view(sequence.size()[0], -1)
        hidden = self.fc(embedded).unsqueeze(0)
        if self.bidirectional:
            hidden = hidden.view(2, hidden.size()[1], -1)

        data, labels, char_data = data_manager.batch_sequence(batch)
        data = self.embedding(data)
        data = self.drop(data)
        lstm_out, hidden = self.lstm(data, hidden)

        # from output of lstm to a fc layer to map to tag space
        tag_space = self.to_tag_space(lstm_out.unsqueeze(1).contiguous())
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
