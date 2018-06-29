import torch
import torch.nn as nn
import torch.nn.utils.rnn as R
from torch.autograd import Variable

import data_manager


# lstm-crf implementation, heavily inspired by the pytorch tutorial and by kaniblu@github

def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))


class CRF(nn.Module):
    def __init__(self, vocab_size):
        super(CRF, self).__init__()

        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.tagset_size = self.n_labels
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels), requires_grad=True)

    def forward(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            # expand tag scores over columns
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            # expand score of tags at previous time step over rows
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            # expand transitions over batchs
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)

            # obtain scores of tags for this step
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            # update alpha, get alpha of current step + carry over alphas of sentences that have already been finished
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        # last step
        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[self.stop_idx].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(list(reversed(pointers)))
        scores, idx = vit.max(1, keepdim=True)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in pointers:
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)
            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = labels.data.new(batch_size, seq_len + 2)
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels

        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1, lens.max() + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score


class LstmCrf(nn.Module):
    def __init__(self, w2v_weights, tag_to_itx, hidden_dim, drop_rate, bidirectional=False, freeze=True,
                 embedding_norm=6):

        super(LstmCrf, self).__init__()

        self.vocab_size = w2v_weights.shape[0]
        self.embedding_dim = w2v_weights.shape[1]

        # self.n_feats = len(word_dims)
        # self.total_word_dim = sum(word_dims)
        # self.word_dims = word_dims
        self.hidden_dim = hidden_dim
        self.dropout_prob = drop_rate

        self.crf = CRF(len(tag_to_itx))
        self.bidirectional = bidirectional
        self.n_labels = self.crf.n_labels

        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights).cuda(), freeze=freeze)
        self.embeddings.max_norm = embedding_norm
        self.drop = nn.Dropout(self.dropout_prob)

        self.output_hidden_dim = self.hidden_dim

        self.bnorm = nn.BatchNorm2d(1)
        self.bnorm2 = nn.BatchNorm2d(1)
        self.fc = nn.Linear(self.hidden_dim, self.n_labels)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim // (1 if not bidirectional else 2),
                            num_layers=1,
                            bidirectional=self.bidirectional,
                            batch_first=True)

        if bidirectional:
            self.init_hidden = self.__init_hidden_bidirectional

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the lstm layer.
        :return: Initialized hidden state of the lstm.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim).cuda(),
                torch.zeros(1, batch_size, self.hidden_dim).cuda())

    def __init_hidden_bidirectional(self, batch_size):
        """
        Inits the hidden state of the lstm layer.
        :return: Initialized hidden state of the lstm.
        """
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, batch_size, self.hidden_dim // 2).cuda(),
                torch.zeros(2, batch_size, self.hidden_dim // 2).cuda())

    def _forward_bilstm(self, data, lengths):
        # n_feats, batch_size, seq_len = xs.size()
        batch_size, seq_len = data.size()

        embedded = self.embeddings(data)
        embedded = embedded.view(batch_size, seq_len, self.embedding_dim)
        embedded = self.drop(embedded)

        # pack, pass through lstm, unpack
        packed = R.pack_padded_sequence(embedded, sorted(lengths.data.tolist(), reverse=True), batch_first=True)
        hidden = self.init_hidden(batch_size)
        output, _ = self.lstm(packed, hidden)
        o, lengths = R.pad_packed_sequence(output, batch_first=True)

        # pass through fc layer and activation
        o = o.contiguous()
        o = self.bnorm(o.unsqueeze(1)).squeeze(1)
        o = self.fc(o)
        o = self.bnorm2(o.unsqueeze(1)).squeeze(1)
        return o

    def _bilstm_score(self, feats, labels, lens):
        batch_size, max_length = labels.size()
        y_exp = labels.unsqueeze(-1)
        labels[labels == -1] = 0
        scores = torch.gather(feats, 2, y_exp).squeeze(-1)
        mask = sequence_mask(lens, max_length).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

    def score(self, feats, labels, lengths):
        transition_score = self.crf.transition_score(labels, lengths)
        bilstm_score = self._bilstm_score(feats, labels, lengths)

        score = transition_score + bilstm_score
        return score

    def forward(self, batch):
        data, labels, _ = data_manager.batch_sequence(batch)
        lengths = self.get_lengths(labels)

        feats = self._forward_bilstm(data, lengths)
        scores, predictions = self.crf.viterbi_decode(feats, lengths)

        # pad predictions so that they match in length with padded labels
        batch_size, pad_to = labels.size()
        _, pad_from = predictions.size()
        padding = torch.zeros(batch_size, pad_to - pad_from).long().cuda()
        predictions = torch.cat([predictions, padding], dim=1)
        predictions = predictions.expand(*labels.size())
        predictions[predictions == 43] = 0
        predictions[predictions == 44] = 0

        return predictions.view(-1), labels.view(-1)

    def get_lengths(self, labels, padding=-1):
        batchs, _ = labels.size()
        lengths = torch.zeros(batchs).long().cuda()
        for i in range(batchs):
            while len(labels[i]) > lengths[i] and labels[i][lengths[i]] != padding:
                lengths[i] += 1
        return lengths

    def get_labels(self, labels, padding=-1):
        tmp = labels != padding
        tmp = torch.sum(tmp, dim=1)
        max_length = torch.max(tmp)
        res = labels[:, :max_length]
        return res

    def neg_log_likelihood(self, batch):
        data, labels, _ = data_manager.batch_sequence(batch)
        lengths = self.get_lengths(labels)
        labels = self.get_labels(labels)

        # get feats from lstm
        feats = self._forward_bilstm(data, lengths)

        # get score of sentence from model
        norm_score = self.crf(feats, lengths)

        # get score of labelling
        sequence_score = self.score(feats, labels, lengths)

        loglik = sequence_score - norm_score
        loglik = -loglik.mean()

        return loglik


def sequence_mask(lens, max_len):
    batch_size = lens.size(0)

    ranges = torch.arange(0, max_len).long().cuda()
    ranges = ranges.expand(batch_size, -1)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    # set 1 where [batch][i] with i < length of phrase
    mask = ranges < lens_exp
    return mask
