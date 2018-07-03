import torch
import torch.nn as nn
from torch.autograd import Variable

import data_manager


# recurrent-crf implementation, heavily inspired by the pytorch tutorial and by kaniblu@github


class CRF(nn.Module):
    def __init__(self, vocab_size):
        super(CRF, self).__init__()

        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.tagset_size = self.n_labels
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels), requires_grad=True)

    @staticmethod
    def log_sum_exp(vec, dim=0):
        """
        Numerically stable log sum exp.
        :param vec: Vector of values.
        :param dim: Dimension over which the log sum exp is being done.
        :return: Log sum exp value/s.
        """
        max, idx = torch.max(vec, dim)
        max_exp = max.unsqueeze(-1).expand_as(vec)
        return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

    def forward(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, tagset_size] FloatTensor
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
            alpha_nxt = self.log_sum_exp(mat, 2).squeeze(-1)

            # update alpha, get alpha of current step + carry over alphas of sentences that have already been finished
            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        # last step
        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = self.log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, tagset_size] FloatTensor
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

    def transition_score(self, labels, lengths):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lengths: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = labels.data.new(batch_size, seq_len + 2)
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels

        mask = sequence_mask(lengths + 1, max_len=seq_len + 2).long()
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

        mask = sequence_mask(lengths + 1, lengths.max() + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score


class LstmCrf(nn.Module):
    def __init__(self, w2v_weights, tag_to_itx, hidden_dim, drop_rate, bidirectional=False, freeze=True,
                 embedding_norm=6):

        super(LstmCrf, self).__init__()

        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_itx)
        self.embedding_dim = w2v_weights.shape[1]
        self.w2v_weights = w2v_weights
        self.bidirectional = bidirectional

        self.drop_rate = drop_rate
        self.drop = nn.Dropout(self.drop_rate)

        # embedding layer
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(w2v_weights), freeze=freeze)
        self.embeddings.max_norm = embedding_norm

        # recurrent and mapping to tagset
        self.recurrent = nn.LSTM(input_size=self.embedding_dim,
                                 hidden_size=self.hidden_dim // (1 if not self.bidirectional else 2),
                                 bidirectional=self.bidirectional, batch_first=True)
        self.bnorm = nn.BatchNorm2d(1)
        self.fc = nn.Linear(self.hidden_dim, self.tagset_size + 2)  # + 2 because of start and end token
        self.bnorm2 = nn.BatchNorm2d(1)

        # crf for scoring at a global level
        self.crf = CRF(self.tagset_size)

    @staticmethod
    def features_score(feats, labels, lengths):
        """
        Given the label scores (feats) of each token and the correct labels,
        return the score of the whole sentence.
        :param feats: Label scores for each token, size = (batch, sentence length, tagset size)
        :param labels: Correct label of each word, size = (batch, sentence length)
        :param lengths: Lengths of each sentence, needed for masking out padding. size = (batch)
        :return: Score of each sentence, size = (batch)
        """
        batch_size, max_length = labels.size()
        labels_exp = labels.unsqueeze(-1)
        # set paddings to 0
        labels[labels == -1] = 0
        # get the score that was given to each correct label
        scores = torch.gather(feats, 2, labels_exp).squeeze(-1)
        # mask out scores of padding
        mask = sequence_mask(lengths, max_length).float()
        scores = scores * mask

        # sum and return
        score = scores.sum(1).squeeze(-1)
        return score

    @staticmethod
    def get_labels(labels, padding=-1):
        """
        Get labels of each sentence, keeping only as much as needed (up to the length of the longest sentence).
        :param labels: Labels of each word for each sentence.
        :param padding: Padding value to use, default -1.
        :return: Labels of each sentence, size = (batch, longest sentence).
        """
        tmp = labels != padding
        tmp = torch.sum(tmp, dim=1)
        max_length = torch.max(tmp)
        res = labels[:, :max_length]
        return res

    def init_hidden(self, batch_size):
        """
        Inits the hidden state of the recurrent layer.
        :param batch_size
        :return: Initialized hidden state of the recurrent layer.
        """
        if self.bidirectional:
            state = [torch.zeros(self.recurrent.num_layers * 2, batch_size, self.hidden_dim // 2),
                     torch.zeros(self.recurrent.num_layers * 2, batch_size, self.hidden_dim // 2)]
        else:
            state = [torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_dim),
                     torch.zeros(self.recurrent.num_layers, batch_size, self.hidden_dim)]
        if next(self.parameters()).is_cuda:
            state[0] = state[0].cuda()
            state[1] = state[1].cuda()
        return state

    def get_features_from_recurrent(self, data, lengths):
        """
        For each word get its scores for each possible label.
        :param data: Input sentences.
        :param lengths: Lengths of each sentence, needed for packing.
        :return: Labels scores of each token, size = (batch, sentence length, tagset size)
        """
        # n_feats, batch_size, seq_len = xs.size()
        batch_size, seq_len = data.size()

        # embed and drop
        embedded = self.embeddings(data)
        embedded = embedded.view(batch_size, seq_len, self.embedding_dim)
        embedded = self.drop(embedded)

        # pack, pass through recurrent, unpack
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted(lengths.data.tolist(), reverse=True),
                                                   batch_first=True)
        hidden = self.init_hidden(batch_size)
        output, _ = self.recurrent(packed, hidden)
        o, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # pass through fc layer and activation
        o = o.contiguous()
        o = self.bnorm(o.unsqueeze(1)).squeeze(1)
        o = self.fc(o)
        o = self.bnorm2(o.unsqueeze(1)).squeeze(1)
        return o

    def score(self, feats, labels, lengths):
        """
        Score a sentence given scores from the recurrent layer and transition scores from the crf.
        :param feats: Label scores for each token, size = (batch, sentence length, tagset size)
        :param labels: Correct label of each word, size = (batch, sentence length)
        :param lengths: Lengths of each sentence, needed for masking out padding. size = (batch)
        :return: Sentence score, size = (batch)
        """
        transition_score = self.crf.transition_score(labels, lengths)
        bilstm_score = self.features_score(feats, labels, lengths)

        score = transition_score + bilstm_score
        return score

    def forward(self, batch):
        data, labels, _ = data_manager.batch_sequence(batch)
        lengths = self.get_lengths(labels)

        # get features and do predictions maximizing the sentence score using the crf
        feats = self.get_features_from_recurrent(data, lengths)
        scores, predictions = self.crf.viterbi_decode(feats, lengths)

        # pad predictions so that they match in length with padded labels
        batch_size, pad_to = labels.size()
        _, pad_from = predictions.size()
        padding = torch.zeros(batch_size, pad_to - pad_from).long()
        if next(self.parameters()).is_cuda:
            padding = padding.cuda()
        predictions = torch.cat([predictions, padding], dim=1)
        predictions = predictions.expand(*labels.size())

        # remove start and stop tags if there are any (mostly for safety, should not happen)
        predictions[predictions == 43] = 0
        predictions[predictions == 44] = 0

        return predictions.view(-1), labels.view(-1)

    def get_lengths(self, labels, padding=-1):
        """
        Get length of each sentences.
        :param labels: Labels of each word for each sentence.
        :param padding: Padding value to use, default -1.
        :return: Length of each sentence, size = (batch).
        TODO: remove for cycle, make it with matrix operations
        """
        batchs, _ = labels.size()
        lengths = torch.zeros(batchs).long()
        if next(self.parameters()).is_cuda:
            lengths = lengths.cuda()
        for i in range(batchs):
            while len(labels[i]) > lengths[i] and labels[i][lengths[i]] != padding:
                lengths[i] += 1
        return lengths

    def neg_log_likelihood(self, batch):
        """
        Used for training, returns a loss that depends on the difference between the score that the model
        would give to the sentence annd the score it would give to the correct labeling of the sentence.
        :param batch:
        :return:Pytorch loss.
        """
        data, labels, _ = data_manager.batch_sequence(batch)
        lengths = self.get_lengths(labels)
        labels = self.get_labels(labels)

        # get feats (scores for each label, for each word) from recurrent
        feats = self.get_features_from_recurrent(data, lengths)
        # get score of sentence from crf
        norm_score = self.crf(feats, lengths)

        # get score that the model would give to the correct labels
        sequence_score = self.score(feats, labels, lengths)

        loglik = sequence_score - norm_score
        loglik = -loglik.mean()
        return loglik


def sequence_mask(lengths, max_len):
    batch_size = lengths.size(0)

    ranges = torch.arange(0, max_len).long()
    if lengths.is_cuda:
        ranges = ranges.cuda()
    ranges = ranges.expand(batch_size, -1)
    lens_exp = lengths.unsqueeze(1).expand_as(ranges)
    # set 1 where [batch][i] with i < length of phrase
    mask = ranges < lens_exp
    return mask
