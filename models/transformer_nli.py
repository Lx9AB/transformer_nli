import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP.embeddings import StaticEmbedding, CNNCharEmbedding, StackEmbedding
from torch.autograd import Variable

from models.utils import LayerNorm
import global_value as gol


class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, k_dim=None, v_dim=None, bias=True, dropout=0.0, nsa=False):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_dim = embed_dim // num_heads if k_dim is None else k_dim
        self.v_dim = embed_dim // num_heads if v_dim is None else v_dim
        self.nsa = nsa

        self.bias = bias

        self.scaling = self.k_dim ** -0.5

        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_heads * self.k_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_heads * self.v_dim, bias=bias)

        self.out_proj = nn.Linear(self.num_heads * self.v_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.bias:
            nn.init.constant_(self.q_proj.bias, 0.)
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask=None):
        # len, bsz, embed_dim
        bsz = query.size(1)
        tgt_len = query.size(0)
        src_len = value.size(0)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling

        q = q.view(-1, bsz * self.num_heads, self.k_dim).transpose(0, 1)
        k = k.view(-1, bsz * self.num_heads, self.k_dim).transpose(0, 1)
        v = v.view(-1, bsz * self.num_heads, self.v_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            mask = mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if self.nsa:
            epoch = gol.get_value("epoch")
            epoch = 50 if epoch is None else epoch
            epoch = 50 if epoch > 50 else epoch
            d = math.log(1000 / epoch - 1)
            if self.training:
                noise = -torch.empty_like(attn_weights).exponential_().log()
                attn_weights_noise = attn_weights + noise
                attn_weights_max = attn_weights_noise.max(dim=-1, keepdim=True)[0]
                mask = attn_weights_noise < (attn_weights_max - d)
            else:
                attn_weights_max = attn_weights.max(dim=-1, keepdim=True)[0]
                mask = attn_weights < (attn_weights_max - d)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads * self.v_dim)

        attn = self.out_proj(attn)

        return attn, attn_weights


class PositionwiseFeedForward(nn.Module):

    def __init__(self, embed_dim, ffn_embed_dim, bias=True, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim, bias=bias)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim, bias=bias)
        self.bias = bias
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.bias:
            nn.init.constant_(self.fc1.bias, 0.)
            nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, ffn_embed_dim, num_heads, k_dim=None, v_dim=None,
                 attention_dropout=0., activation_dropout=0., dropout=0.):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=attention_dropout
        )
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.ffn = PositionwiseFeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            dropout=activation_dropout
        )
        self.ffn_layer_norm = LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, x, mask=None):
        residual = x
        x, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.self_attn_layer_norm(residual + x)

        residual = x
        x = self.ffn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ffn_layer_norm(residual + x)

        return x


class TransformerInteractionLayer(nn.Module):

    def __init__(self, embed_dim, ffn_embed_dim, num_heads, k_dim=None, v_dim=None,
                 attention_dropout=0., activation_dropout=0., dropout=0.):
        super(TransformerInteractionLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=attention_dropout
        )
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.encoder_attn = MultiHeadedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_dim=k_dim,
            v_dim=v_dim,
            dropout=attention_dropout
        )
        self.encoder_attn_layer_norm = LayerNorm(embed_dim)
        self.ffn = PositionwiseFeedForward(
            embed_dim=embed_dim,
            ffn_embed_dim=ffn_embed_dim,
            dropout=activation_dropout
        )
        self.ffn_layer_norm = LayerNorm(embed_dim)
        self.dropout = dropout

    def forward(self, x_1, x_2, mask_1, mask_2):
        residual = x_1
        x_1, _ = self.self_attn(query=x_1, key=x_1, value=x_1, mask=mask_1)
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)
        x_1 = self.self_attn_layer_norm(residual + x_1)

        residual = x_2
        x_2, _ = self.self_attn(query=x_2, key=x_2, value=x_2, mask=mask_2)
        x_2 = F.dropout(x_2, p=self.dropout, training=self.training)
        x_2 = self.self_attn_layer_norm(residual + x_2)

        residual_1 = x_1
        residual_2 = x_2
        y_1, _ = self.encoder_attn(query=x_1, key=x_2, value=x_2, mask=mask_2)
        y_1 = F.dropout(y_1, p=self.dropout, training=self.training)
        y_2, _ = self.encoder_attn(query=x_2, key=x_1, value=x_1, mask=mask_1)
        y_2 = F.dropout(y_2, p=self.dropout, training=self.training)
        x_1 = self.encoder_attn_layer_norm(residual_1 + y_1)
        x_2 = self.encoder_attn_layer_norm(residual_2 + y_2)

        residual = x_1
        x_1 = self.ffn(x_1)
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)
        x_1 = self.ffn_layer_norm(residual + x_1)

        residual = x_2
        x_2 = self.ffn(x_2)
        x_2 = F.dropout(x_2, p=self.dropout, training=self.training)
        x_2 = self.ffn_layer_norm(residual + x_2)

        return x_1, x_2


class TransformerEncoder(nn.Module):

    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.N = args.N
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(
            embed_dim=args.embed_dim,
            ffn_embed_dim=args.ffn_embed_dim,
            num_heads=args.num_heads,
            k_dim=args.k_dim,
            v_dim=args.v_dim,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            dropout=args.dropout) for _ in range(self.N)])

    def forward(self, x, mask=None):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


class TransformerInteraction(nn.Module):

    def __init__(self, args):
        super(TransformerInteraction, self).__init__()
        self.M = args.M
        self.decoder_layers = nn.ModuleList([TransformerInteractionLayer(
            embed_dim=args.embed_dim,
            ffn_embed_dim=args.ffn_embed_dim,
            num_heads=args.num_heads,
            k_dim=args.k_dim,
            v_dim=args.v_dim,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            dropout=args.dropout) for _ in range(self.M)])

    def forward(self, x_1, x_2, mask_1=None, mask_2=None):
        for layer in self.decoder_layers:
            x_1, x_2 = layer(x_1, x_2, mask_1=mask_1, mask_2=mask_2)
        return x_1, x_2


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., embed_dim, 2) *
                             -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0), :], requires_grad=False)
        return x


class Comparison(nn.Module):

    def __init__(self, args, label_vocab):
        super(Comparison, self).__init__()
        self.fc1 = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.fc2 = nn.Linear(args.embed_dim, args.embed_dim)
        self.fc3 = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.fc4 = nn.Linear(args.embed_dim, len(label_vocab))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        if self.bias:
            nn.init.constant_(self.fc1.bias, 0.)
            nn.init.constant_(self.fc2.bias, 0.)
            nn.init.constant_(self.fc3.bias, 0.)
            nn.init.constant_(self.fc4.bias, 0.)

    def forward(self, encoding_1, encoding_2, interaction_1, interaction_2, mask_1=None, mask_2=None):
        x_1 = torch.cat([encoding_1, interaction_1], dim=-1)
        x_1 = self.fc2(torch.relu(self.fc1(x_1)))
        if mask_1 is not None:
            mask_1 = (mask_1 == 0).transpose(0, 1).transpose(0, 2).type_as(x_1)
            x_1 = (x_1 * mask_1).sum(dim=0) / ((mask_1 == 0).sum(dim=0) ** 0.5)
        else:
            x_1 = x_1.sum(dim=0) / (x_1.size(0) ** 0.5)

        x_2 = torch.cat([encoding_2, interaction_2], dim=-1)
        x_2 = self.fc2(torch.relu(self.fc1(x_2)))
        if mask_2 is not None:
            mask_2 = (mask_2 == 0).transpose(0, 1).transpose(0, 2).type_as(x_2)
            x_2 = (x_2 * mask_2).sum(dim=0) / ((mask_2 == 0).sum(dim=0) ** 0.5)
        else:
            x_2 = x_2.sum(dim=0) / (x_2.size(0) ** 0.5)

        x = torch.cat([x_1, x_2], dim=-1)
        x = self.fc4(torch.relu(self.fc3(x)))
        return torch.log_softmax(x, dim=-1)


class Embedding(nn.Module):

    def __init__(self, args, vocab):
        super(Embedding, self).__init__()
        self.embed_scale = args.embed_dim ** 0.5
        word_embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-840B-300d', lower=True,
                                     requires_grad=False, min_freq=20,
                                     only_use_pretrain_word=True, only_train_min_freq=True)
        char_embed = CNNCharEmbedding(vocab, embed_size=30)
        self.embed = StackEmbedding([word_embed, char_embed])
        self.lut_proj = nn.Linear(self.embed.embed_size, args.embed_dim, bias=False)
        self.pe = PositionalEncoding(args.embed_dim, max_len=args.max_len)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lut_proj.weight)
        if self.bias:
            nn.init.constant_(self.lut_proj.bias, 0.)

    def forward(self, x):
        x = self.lut_proj(self.embed(x)).transpose(0, 1)
        x = x * self.embed_scale
        x = self.pe(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerNLI(nn.Module):

    def __init__(self, args, vocab, label_vocab):
        super(TransformerNLI, self).__init__()
        self.embedding = Embedding(args, vocab)
        self.encoder = TransformerEncoder(args)
        self.interaction = TransformerInteraction(args)
        self.comparison = Comparison(args, label_vocab)

    def forward(self, premise, hypothesis):
        premise_mask = (premise == 0).unsqueeze(1)
        hypothesis_mask = (hypothesis == 0).unsqueeze(1)
        premise = self.embedding(premise)
        hypothesis = self.embedding(hypothesis)

        encoding_1 = self.encoder(premise, mask=premise_mask)
        encoding_2 = self.encoder(hypothesis, mask=hypothesis_mask)

        interaction_1, interaction_2 = self.interaction(encoding_1, encoding_2, mask_1=premise_mask,
                                                        mask_2=hypothesis_mask)
        y = self.comparison(encoding_1, encoding_2, interaction_1, interaction_2, premise_mask, hypothesis_mask)
        return y
