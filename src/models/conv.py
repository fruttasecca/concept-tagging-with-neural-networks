import torch
import torch.nn as nn

import data_manager


class CONV(nn.Module):
    def __init__(self, device, w2v_weights, hidden_dim, tagset_size, pad_sentence_length, drop_rate=0.5, bidirectional=False,
                 freeze=True, embedding_norm=10):
        """
        :param device: Device to which to map tensors (GPU or CPU).
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param hidden_dim The hidden memory of the recurrent layer will have a size of 3 times this.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param pad_sentence_length: Max length of a padded sentence (needed for convolution).
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the recurrent should be bidirectional.
        :param freeze: If the embedding parameters should be frozen or trained during training.
        :param embedding_norm: Max norm of the embeddings.
        """
        super(CONV, self).__init__()

        self.device = device
        self.tagset_size = tagset_size
        self.embedding_dim = w2v_weights.shape[1]
        self.pad_sentence_length = pad_sentence_length
        self.w2v_weights = w2v_weights
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
        self.embedding.max_norm = embedding_norm

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)

        # channels outputted by conv networks
        self.feats = hidden_dim

        # conv layer for single word, bigram, trigram
        self.ngram1 = nn.Sequential(
            nn.Conv2d(1, self.feats, kernel_size=(1, self.embedding_dim), stride=(1, self.embedding_dim), padding=0),
            nn.Dropout2d(p=self.drop_rate),
            nn.BatchNorm2d(self.feats),
            nn.MaxPool2d(kernel_size=(self.pad_sentence_length, 1)),
            nn.ReLU(inplace=True)
        )

        self.ngram2 = nn.Sequential(
            nn.Conv2d(1, self.feats, kernel_size=(2, self.embedding_dim), stride=(1, self.embedding_dim), padding=0),
            nn.Dropout2d(p=self.drop_rate),
            nn.BatchNorm2d(self.feats),
            nn.MaxPool2d(kernel_size=(self.pad_sentence_length - 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.ngram3 = nn.Sequential(
            nn.Conv2d(1, self.feats, kernel_size=(3, self.embedding_dim), stride=(1, self.embedding_dim), padding=0),
            nn.Dropout2d(p=self.drop_rate),
            nn.BatchNorm2d(self.feats),
            nn.MaxPool2d(kernel_size=(self.pad_sentence_length - 2, 1)),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.GRU(self.embedding_dim, (self.feats * 3) // (1 if not bidirectional else 2),
                           batch_first=True, bidirectional=bidirectional)

        self.to_tag_space = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.feats * 3, self.tagset_size),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def prepare_batch(self, batch):
        seq_list = []
        for sample in batch:
            seq = sample["sequence_extra"].unsqueeze(1)
            seq_list.append(seq.to(self.device))
        res_seq = torch.cat(seq_list, dim=0)
        return res_seq

    def forward(self, batch):
        sentence_as_matrix = self.prepare_batch(batch)

        embedded = self.embedding(sentence_as_matrix)
        embedded = self.drop(embedded)

        # convolution on data
        n1 = self.ngram1(embedded)
        n2 = self.ngram2(embedded)
        n3 = self.ngram3(embedded)

        # combine result in a vector that will be the initial hidden state of the recurrent layer
        batch_size = embedded.size()[0]
        n1 = n1.view(batch_size, -1)
        n2 = n2.view(batch_size, -1)
        n3 = n3.view(batch_size, -1)
        hidden = torch.cat((n1, n2, n3), dim=1).unsqueeze(0)
        if self.bidirectional:
            hidden = hidden.view(2, hidden.size()[1], -1)

        data, labels, _ = data_manager.batch_sequence(batch, self.device)
        data = self.embedding(data)
        data = self.drop(data)
        lstm_out, hidden = self.lstm(data, hidden)

        # send output to fc layer(s)
        tag_space = self.to_tag_space(lstm_out.unsqueeze(1).contiguous())
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
