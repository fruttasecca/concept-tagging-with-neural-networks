import torch
import torch.nn as nn
import torch.nn.functional as F

import data_manager


class Attention(nn.Module):

    def __init__(self, w2v_weights, decoder_embedding_size, hidden_dim, tagset_size, drop_rate=0.5, bidirectional=False,
                 freeze=True, max_norm_emb1=10, max_norm_emb2=1, padded_sentence_length=25):
        """
        :param w2v_weights: Matrix of w2v w2v_weights, ith row contains the embedding for the word mapped to the ith index, the
        last row should correspond to the padding token, <padding>.
        :param decoder_embedding_size Size of the learned embeddings of the tags.
        :param hidden_dim Size of the hidden dimension of the lstm layer.
        :param tagset_size: Number of possible classes, this will be the dimension of the output vector.
        :param drop_rate: Drop rate for regularization.
        :param bidirectional: If the lstm should be bidirectional.
        :param freeze: If the embedding parameters should be frozen or trained during training.
        :param max_norm_emb1 Max norm of the embeddings of tokens (used by the encoder), default 10.
        :param max_norm_emb2 Max norm of the embeddings of tags (used by the decoder), default 10.
        :param padded_sentence_length Length to which sentences are padded.
        """
        super(Attention, self).__init__()

        self.w2v_weights = w2v_weights
        self.hidden_dim = hidden_dim
        self.embedding_dim = w2v_weights.shape[1]
        self.decoder_embedding_size = decoder_embedding_size
        self.tagset_size = tagset_size
        self.drop_rate = drop_rate
        self.max_length = padded_sentence_length  # max length over any phrase
        self.bidirectional = bidirectional

        self.drop = nn.Dropout(self.drop_rate)
        # encoder section, gru layer accepts inputs of size hiddendim and as hidden state of the same shape
        # embeddings for the input tokens
        self.embedding_encoder = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights).cuda(), freeze=freeze)
        self.embedding_encoder.max_norm = max_norm_emb1
        self.gru_encoder = nn.GRU(self.embedding_dim, self.hidden_dim // (1 if not bidirectional else 2),
                                  batch_first=True, bidirectional=bidirectional, dropout=drop_rate)

        if bidirectional:
            self.init_hidden_encoder = self.__init_hidden_encoder_bidirectional

        # decoder section
        # embeddings that are going to represent the tags, +1 for padding index
        self.embedding_decoder = nn.Embedding(self.tagset_size + 1, self.decoder_embedding_size, max_norm=max_norm_emb2)
        self.gru_decoder = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True, dropout=drop_rate)
        self.bnorm = nn.BatchNorm2d(1)
        self.out = nn.Linear(self.hidden_dim, self.tagset_size)
        self.softmax = nn.LogSoftmax(dim=2)

        # attention related layers
        self.attn = nn.Linear(self.decoder_embedding_size + self.hidden_dim, self.max_length)
        self.attn_combine = nn.Linear(self.decoder_embedding_size + self.hidden_dim, self.hidden_dim)
        self.bnorm2 = nn.BatchNorm2d(1)

    def init_hidden_encoder(self, batch_size):
        """
        Inits the hidden state of the gru layer.
        :param batch_size
        :return: Initialized hidden state of the gru encoder.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(self.gru_encoder.num_layers, batch_size, self.hidden_dim).cuda()

    def __init_hidden_encoder_bidirectional(self, batch_size):
        """
        Inits the hidden state of the gru layer.
        :param batch_size
        :return: Initialized hidden state of the gru encoder.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return torch.zeros(self.gru_encoder.num_layers * 2, batch_size, self.hidden_dim // 2).cuda()

    def decoder_forward(self, input, hidden, encoder_outputs):
        """
        Forward function of the decoder section, meant to be used on inputs of a sequence being passed one at a time.
        :param input: Input of the form (batch, 1, hidden dim).
        :param hidden: Hidden state from a previous iteration.
        :param encoder_outputs: Outputs of the encoder, to be used with attention.
        :return: Output for the current token of size (batch, tagset) and a new hidden state.
        """
        output = self.embedding_decoder(input)
        output = self.drop(output)

        lookat = torch.cat((output, hidden.squeeze(0).unsqueeze(1)), dim=2)
        # softmax so that they sum to one
        attn_weights = F.softmax(self.attn(lookat), dim=2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        output = torch.cat((output, attn_applied), 2)
        output = self.attn_combine(output)
        output = self.bnorm2(output.unsqueeze(1))
        output = self.drop(output)
        output = output.squeeze(1)

        output = F.relu(output)
        output, hidden = self.gru_decoder(output, hidden)
        output = self.bnorm(output.unsqueeze(1))
        output = output.squeeze(1)
        output = self.drop(output)
        output = self.out(output)
        output = self.softmax(output)
        return output, hidden

    def forward(self, batch):
        # init hidden layers for both encoder and decoder section
        hidden_encoder = self.init_hidden_encoder(len(batch))

        # pack data into a batch
        data, labels, char_data = data_manager.batch_sequence(batch)
        data = self.embedding_encoder(data)
        data = self.drop(data)

        # encode
        encoder_output, hidden_encoder = self.gru_encoder(data, hidden_encoder)

        # set first token passed to decoder as tagset_size, mapped to the last row of the embedding
        decoder_input = torch.zeros(len(batch), 1).long().cuda()
        torch.add(decoder_input, self.tagset_size, decoder_input)  # special start character

        # encoder will pass its hidden state to the decoder
        if self.bidirectional:  # needs to be reshaped since a bidirectiona layer will return (2, batch, hidden dim//2)
            hidden_decoder = torch.cat((hidden_encoder[0], hidden_encoder[1]), dim=1).unsqueeze(0)
        else:
            hidden_decoder = hidden_encoder

        results = []
        # decode
        for di in range(encoder_output.size()[1]):  # max length of any phrase in the batch
            decoder_output, hidden_decoder = self.decoder_forward(decoder_input, hidden_decoder, encoder_output)

            _, topi = decoder_output.topk(1)  # extract predicted label
            decoder_input = topi.squeeze(1).detach().cuda()  # detach from history as input
            results.append(decoder_output)

        results = torch.cat(results, dim=1)
        return results.view(-1, self.tagset_size), labels.view(-1)
