import torch
import logging
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class RNN(nn.Module):
    def __init__(self, args, tokenizer):
        super(RNN, self).__init__()

        self.args = args
        self.vocab_size = len(tokenizer)

        self.embedding = nn.Embedding(self.vocab_size, self.args.embedding_dim)

        self.rnn = nn.RNN(self.args.embedding_dim, self.args.hid_dim)

        self.fc = nn.Linear(self.args.hid_dim, 2)

    def forward(self, inputs):
        # text = [sent len, batch size]

        embedded = self.embedding(inputs.permute(1, 0))

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class Low_LSTM(nn.Module):
    def __init__(self, args, tokenizer):
        super(Low_LSTM, self).__init__()
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args
        self.dropout_r = args.dropout
        self.dropout = nn.Dropout(self.dropout_r)

        self.embedding = nn.Embedding(self.vocab_size, self.args.embedding_dim, padding_idx=self.pad_ids)

        self.rnn = nn.LSTM(self.args.embedding_dim,
                           self.args.hid_dim,
                           num_layers=1,
                           bidirectional=False,
                           dropout=self.dropout_r)

        self.fc = nn.Linear(self.args.hid_dim, 2)

    def forward(self,
                inputs) -> Tensor:

        embedded = self.dropout(self.embedding(inputs.permute(1, 0)))

        output, (hidden, cell) = self.rnn(embedded)

        hidden = self.dropout(hidden.squeeze(0))

        return self.fc(hidden)


class LSTM(nn.Module):
    def __init__(self, args, tokenizer):
        super(LSTM, self).__init__()
        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)

        self.args = args
        self.dropout_r = args.dropout
        self.dropout = nn.Dropout(self.dropout_r)

        self.embedding = nn.Embedding(self.vocab_size, self.args.embedding_dim, padding_idx=self.pad_ids)

        self.rnn = nn.LSTM(self.args.embedding_dim,
                           self.args.hid_dim,
                           num_layers=1,
                           bidirectional=True,
                           dropout=self.dropout_r)

        self.fc = nn.Linear(self.args.hid_dim * 2, 2)

    def forward(self,
                inputs) -> Tensor:

        embedded = self.dropout(self.embedding(inputs.permute(1, 0)))

        output, (hidden, cell) = self.rnn(embedded)

        bidirectional_hid_1, bidirectional_hid_h2 = hidden[-2, :, :], hidden[-1, :, :]

        logits = self.dropout(torch.cat([bidirectional_hid_1, bidirectional_hid_h2], dim=1))

        return self.fc(logits)