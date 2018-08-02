import torch
import torch.nn as nn

import data_manager


class FCINIT(nn.Module):
    def __init__(self, device, w2v_weights, hidden_dim, tagset_size, pad_sentence_length, drop_rate, bidirectional=False,
                 freeze=True, embedding_norm=10):
        """
        :param device: Device to which to map tensors (GPU or CPU).
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param hidden_dim The hidden memory of the recurrent layer will have a size of 3 times this.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the recurrent should be bidirectional.
        :param freeze: If the embedding parameters should be frozen or trained during training.
        :param embedding_norm: Max norm of the embeddings.
        """
        super(FCINIT, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.embedding_dim = w2v_weights.shape[1]
        self.w2v_weights = w2v_weights
        self.bidirectional = bidirectional
        self.pad_sentence_length = pad_sentence_length
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
        self.embedding.max_norm = embedding_norm

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)

        # fc layer to elaborate an hidden state from the whole (padded) sentence, seen as the concatenation of the word
        # embeddings
        self.fc = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.pad_sentence_length * self.embedding_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
        )

        self.to_tag_space = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_dim, self.tagset_size),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.recurrent = nn.GRU(self.embedding_dim, self.hidden_dim // (1 if not self.bidirectional else 2),
                                batch_first=True, bidirectional=self.bidirectional)

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the recurrent layer.
        :param batch_size
        :return: Initialized hidden state of the recurrent encoder.
        """
        if self.bidirectional:
            state = torch.zeros(self.recurrent.num_layers * 2, batch_size, self.hidden_dim // 2).to(self.device)
        else:
            state = torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_dim).to(self.device)
        return state

    def prepare_batch(self, batch):
        seq_list = []
        for sample in batch:
            seq = sample["sequence_extra"].unsqueeze(1)
            seq_list.append(seq.to(self.device))
        res_seq = torch.cat(seq_list, dim=0)
        return res_seq

    def forward(self, batch):
        # pre-elaborate hidden state
        sequence = self.prepare_batch(batch)
        embedded = self.embedding(sequence).view(sequence.size()[0], -1)
        hidden = self.fc(embedded).unsqueeze(0)
        if self.bidirectional:
            hidden = hidden.view(2, hidden.size()[1], -1)

        # output scores for each input embedding, use the pre-elaborated hidden state
        data, labels, char_data = data_manager.batch_sequence(batch, self.device)
        data = self.embedding(data)
        data = self.drop(data)
        rec_out, hidden = self.recurrent(data, hidden)

        # from output of the recurrent layer to a fc layer to map to tag space
        tag_space = self.to_tag_space(rec_out.unsqueeze(1).contiguous())
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=3)

        return tag_scores.view(-1, self.tagset_size), labels.view(-1)
