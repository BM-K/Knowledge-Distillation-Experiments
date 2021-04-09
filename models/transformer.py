import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def get_attn_pad_mask(args, seq_q, seq_k, pad_idx):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k).to(args.device)


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, ):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)
        scores.masked_fill_(attn_mask, -1e9)  # fills elements of self tensor with value where mask is one.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiheadAttention(nn.Module):
    def __init__(self, args, t_config):
        super(MultiheadAttention, self).__init__()

        self.args = args
        self.n_heads = t_config['n_heads']
        self.d_model = t_config['d_model']
        self.d_k = int(self.d_model / self.n_heads)
        self.d_v = int(self.d_model / self.n_heads)

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads)  # init (512 x 64 * 8)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads)
        self.li1 = nn.Linear(self.n_heads * self.d_v, self.d_model)

        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s:[batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # k_s:[batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s:[batch_size x n_heads x len_q x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask: [batch_size x n_heads x len_q x len_k]

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, self.d_k)
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.li1(context)

        return self.layer_norm(output + residual), attn
        # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args, t_config):
        super(PoswiseFeedForwardNet, self).__init__()

        self.args = args
        self.d_model = t_config['d_model']
        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.args.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.args.d_ff, out_channels=self.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)

        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, args, t_config):
        super(EncoderLayer, self).__init__()

        self.args = args
        self.enc_self_attn = MultiheadAttention(self.args, t_config)
        self.pos_ffn = PoswiseFeedForwardNet(self.args, t_config)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_inputs to same Q, K, V
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs = [batch_size x len_q x d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, args, t_config, vocab_size, pad_ids):
        super(Encoder, self).__init__()

        self.args = args
        self.pad_ids = pad_ids
        self.d_model = t_config['d_model']
        self.src_emb = nn.Embedding(vocab_size, self.d_model)
        self.pos_embedding = PositionalEncoding(self.d_model, args.max_len)
        self.layers = nn.ModuleList([EncoderLayer(self.args, t_config) for _ in range(t_config['n_layers'])])

    def forward(self, enc_inputs):  # enc_inputs : [batch_size x source_len]
        enc_outputs = self.src_emb(enc_inputs) + self.pos_embedding(enc_inputs)
        enc_self_attn_mask = get_attn_pad_mask(self.args, enc_inputs, enc_inputs, self.pad_ids)  # PAD 에 MASK
        enc_self_attns = []

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Transformer(nn.Module):
    def __init__(self, args, tokenizer, t_config):
        super(Transformer, self).__init__()

        self.pad_ids = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.vocab_size = len(tokenizer)
        self.d_model = t_config['d_model']
        self.encoder = Encoder(args, t_config, self.vocab_size, self.pad_ids)

        self.pooling = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                     nn.Tanh(),
                                     nn.Dropout(p=args.dropout))

        self.linear = nn.Linear(self.d_model, 2)

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        cls_pooling = self.pooling(enc_outputs[:, :1, :])
        outputs = self.linear(cls_pooling.squeeze(1))

        return outputs