import torch
import torch.nn as nn

import data_manager


class CNN(nn.Module):
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
        super(CNN, self).__init__()

        self.pad_sentence_length = pad_sentence_length
        self.tagset_size = tagset_size
        self.w2v_weights = w2v_weights

        self.bidirectional = bidirectional
        self.embedding_dim = w2v_weights.shape[1]

        # channels outputted by conv networks
        self.feats = hidden_dim
        self.conv_drop = drop_rate
        self.fc_drop = drop_rate

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
        self.embedding.max_norm = embedding_norm

        self.drop = nn.Dropout(self.fc_drop)

        self.ngram1 = nn.Sequential(
            nn.Conv2d(1, self.feats, kernel_size=(1, self.embedding_dim), stride=(1, self.embedding_dim), padding=0),
            nn.Dropout2d(p=self.conv_drop),
            nn.BatchNorm2d(self.feats),
            nn.MaxPool2d(kernel_size=(self.pad_sentence_length, 1)),
            nn.ReLU(inplace=True)
        )

        self.ngram2 = nn.Sequential(
            nn.Conv2d(1, self.feats, kernel_size=(2, self.embedding_dim), stride=(1, self.embedding_dim), padding=0),
            nn.Dropout2d(p=self.conv_drop),
            nn.BatchNorm2d(self.feats),
            nn.MaxPool2d(kernel_size=(self.pad_sentence_length - 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.ngram3 = nn.Sequential(
            nn.Conv2d(1, self.feats, kernel_size=(3, self.embedding_dim), stride=(1, self.embedding_dim), padding=0),
            nn.Dropout2d(p=self.conv_drop),
            nn.BatchNorm2d(self.feats),
            nn.MaxPool2d(kernel_size=(self.pad_sentence_length - 2, 1)),
            nn.ReLU(inplace=True)
        )

        self.to_tag_space = nn.Sequential(
            nn.Dropout(self.fc_drop),
            nn.Linear(self.feats * 3, self.tagset_size),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.GRU(self.embedding_dim, (self.feats * 3) // (1 if not bidirectional else 2),
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
        sequence = self.prepare_batch(batch)

        embedded = self.embedding(sequence)
        embedded = self.drop(embedded)
        n1 = self.ngram1(embedded)
        n2 = self.ngram2(embedded)
        n3 = self.ngram3(embedded)

        batch_size = embedded.size()[0]
        n1 = n1.view(batch_size, -1)
        n2 = n2.view(batch_size, -1)
        n3 = n3.view(batch_size, -1)

        hidden = torch.cat((n1, n2, n3), dim=1).unsqueeze(0)
        if self.bidirectional:
            hidden = hidden.view(2, hidden.size()[1], -1)

        data, labels, chars = data_manager.batch_sequence(batch)
        data = self.embedding(data)
        data = self.drop(data)
        lstm_out, hidden = self.lstm(data, hidden)

        # send output to fc layer(s)
        tag_space = self.to_tag_space(lstm_out.unsqueeze(1).contiguous())
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
