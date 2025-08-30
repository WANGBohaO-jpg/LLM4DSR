import pdb
import numpy as np
import torch
import torch.nn as nn
import world


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, item_num):
        super(SASRec, self).__init__()

        # self.user_num = user_num
        self.item_num = item_num
        self.config = world.config

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, self.config["hidden_units"], padding_idx=0)

        self.pos_emb = torch.nn.Embedding(
            self.config["maxlen"], self.config["hidden_units"]
        )  # TO IMPROVE 位置embedding 200个
        self.emb_dropout = torch.nn.Dropout(p=self.config["dropout_rate"])

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)

        for _ in range(self.config["num_blocks"]):
            new_attn_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                self.config["hidden_units"], self.config["num_heads"], self.config["dropout_rate"]
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.config["hidden_units"], self.config["dropout_rate"])
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def log2feats(self, log_seqs, return_attention=False):
        seqs = self.item_emb(log_seqs)  # batch_size x max_len x embedding_dim
        seqs *= self.item_emb.embedding_dim**0.5

        single_seq = torch.arange(log_seqs.shape[1], dtype=torch.long, device="cuda")
        positions = single_seq.unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)  # 使得embedding中某些元素随机归0

        timeline_mask = log_seqs == 0  # batch_size x max_len
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim ；True表示序列中不为padding

        tl = seqs.shape[1]  # time dim len for enforce causality
        # 即判断第i个item是否对第j个item起作用，仅当j>=i时起作用; 返回一个上方为1的上三角矩阵
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device="cuda"))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_output_weights = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)  # query key value
            
            # mha_outputs.shape 10 x 256 x 64
            # attn_output_weights.shape 256 x 10 x 10 最后一行代表序列前面的item对最后一个item的权重

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        
        if return_attention:
            return log_feats, attn_output_weights[:, -1, :]
        else:
            return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # batch_size x max_len x embedding_dim

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)  # bacth_size x maxlen x neg_num x emb_dim

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, log_feats  # pos_pred, neg_pred

    def predict(self, log_seqs):
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :].squeeze(dim=1)  # 返回最后一个item预测的下一个item

        item_embs = self.item_emb.weight
        logits = torch.matmul(final_feat, item_embs.t())
        # item_embs = self.item_emb(item_indices)  # (U, I, C)
        # logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)  # 返回与item库中每个item的匹配程度

        return logits  # preds # (U, I)

    def save_item_embeddings(self, file_path):
        torch.save(self.item_emb.weight.data, file_path)


class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, gru_layers=2):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num + 1

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=gru_layers, batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states):
        # Supervised Head
        sequence_lengths = torch.sum(states != 0, dim=1).cpu()
        emb = self.item_embeddings(states)

        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output

    def predict(self, states):
        # Supervised Head
        sequence_lengths = torch.sum(states != 0, dim=1).cpu()
        emb = self.item_embeddings(states)

        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output
