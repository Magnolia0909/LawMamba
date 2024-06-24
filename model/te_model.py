# -*- coding: utf-8 -*-
import numpy as np
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from .te_similarity import SimilarityAttention

def l1norm(X, dim, eps=1e-8):
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class TextRepresentation(nn.Module):

    def __init__(self, max_seq_length, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, no_txtnorm=False, gpt2_model_name='gpt2'):
        super(TextRepresentation, self).__init__()
        self.max_seq_length = max_seq_length
        self.word_dim = word_dim
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.use_bi_gru = use_bi_gru
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.rnn = nn.GRU(self.word_dim, self.embed_size, num_layers, batch_first=True, bidirectional=self.use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):

        out = self.encode_text(x)
        current_batch, _, _ = out.shape
        len = torch.full((current_batch,), 128, dtype=torch.int64).cuda()

        return out, len

    def encode_text(self, x):
        x = self.embed(x)
        # 通过 GRU 层传递加权嵌入
        cap_emb, _ = self.rnn(x)
        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)//2] + cap_emb[:, :, cap_emb.size(2)//2:])/2
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
        return cap_emb

class GRUModel(nn.Module):

    def __init__(self, args):
        super(GRUModel, self).__init__()
        self.args = args
        self.grad_clip = self.args.grad_clip
        self.device, device_ids = self._prepare_device(self.args.n_gpu)
        self.txt_enc = TextRepresentation(self.args.max_seq_length, self.args.vocab_size, self.args.word_dim,
                                   self.args.embed_size, self.args.num_layers,
                                   use_bi_gru=self.args.bi_gru,
                                   no_txtnorm=self.args.no_txtnorm)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(self.args.embed_size, self.args.num_labels),
        #     nn.Sigmoid()
        # )
        if torch.cuda.is_available():
            self.txt_enc.cuda()
            cudnn.benchmark = True
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.txt_enc.load_state_dict(state_dict[0])

    def train_start(self):
        self.txt_enc.train()

    def test_start(self):
        self.txt_enc.eval()

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def forward(self, contract_inputs, law_inputs_1=None, law_inputs_2=None, law_inputs_3=None):
        if torch.cuda.is_available():
            contracts = contract_inputs.cuda()
            law_inputs_1 = law_inputs_1.cuda()
            law_inputs_2 = law_inputs_2.cuda()
            law_inputs_3 = law_inputs_3.cuda()

        con_emb, _ = self.txt_enc(contracts)
        law_emb_1, _ = self.txt_enc(law_inputs_1)
        law_emb_2, _ = self.txt_enc(law_inputs_2)
        law_emb_3, law_len = self.txt_enc(law_inputs_3)

        return con_emb, law_emb_1, law_emb_2, law_emb_3, law_len
