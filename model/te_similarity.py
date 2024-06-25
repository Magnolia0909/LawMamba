# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from .te_visual import CrossAttentionVisualizer
class SimilarityAttention(nn.Module):
    def __init__(self, args):
        super(SimilarityAttention, self).__init__()
        self.args = args
        self.smooth = self.args.lambda_softmax
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=-1)
        if self.args.model_choice == 'bert' or self.args.model_choice == 'rnn':
            self.embed_dim = self.args.embed_size
        else:
            self.embed_dim = self.args.clip_transformer_width

    def l2norm(self, x, dim):
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        x = x / (norm + 1e-8)
        return x

    def forward(self, contract, law):
        batch_size_c, contractL = contract.size(0), contract.size(1)
        batch_size, lawL = law.size(0), law.size(1)
        contractT = torch.transpose(contract, 1, 2)

        attn = torch.bmm(law.float(), contract.transpose(1, 2).float())
        attn = attn / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        attn = self.leaky_relu(attn)
        attn = self.l2norm(attn, 2)
        attn = F.softmax(attn, dim=-1)
        attn = attn * self.smooth

        attnT = torch.transpose(attn, 1, 2).contiguous()
        lawT = torch.transpose(law, 1, 2)
        weighted_law = torch.bmm(lawT.float(), attnT.float())
        weighted_law = torch.transpose(weighted_law, 1, 2)

        return weighted_law, attn

class ContractToLaw(nn.Module):
    def __init__(self, args, contract=None, law=None):
        super(ContractToLaw, self).__init__()
        self.args = args
        self.simi_attn = SimilarityAttention(self.args)
        if contract:
            self.is_show = True
            self.contract = contract[0]
            self.law = law[0]
        else:
            self.is_show = False

    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    # 创建一个函数来根据注意力权重来标记文本
    def highlight_text(self, tokens, attention_weights):
        highlighted_text = []
        for token, weight in zip(tokens, attention_weights):
            color = plt.cm.viridis(weight)  # 使用与热力图相同的配色
            highlighted_text.append(
                f'<span style="background-color: rgba({color[0] * 255},{color[1] * 255},{color[2] * 255},{color[3]})">{token}</span>')
        return ' '.join(highlighted_text)

    def forward(self, contracts, laws, law_lens):
        similarities = []
        n_contract = contracts.size(0)
        n_law = laws.size(0)
        n_region = contracts.size(1)
        for i in range(n_law):
            n_word = law_lens[i]
            law_i = laws[i, :n_word, :].unsqueeze(0).contiguous()
            law_i_expand = law_i.repeat(n_contract, 1, 1)
            weiLaw, attn = self.simi_attn(contracts, law_i_expand)
            #
            # if self.is_show:
            #     save_path = "/records/attention_weights.txt"
            #     # 将注意力权重保存到文件
            #     contract_attention_weights = attn_show.mean(axis=1)  # 平均每一列的权重
            #     legal_attention_weights = attn_show.mean(axis=0)
            #     with open(save_path, 'w') as f:
            #         f.write(f"contract:{contract_attention_weights}\n")
            #         f.write(f"law:{legal_attention_weights}\n")
                # visualizer = CrossAttentionVisualizer(self.args, self.contract, self.law)
                # visualizer.visualize_attention(attn_show)

            # plt.figure(figsize=(10, 8))
            # sns.heatmap(attn_show, cmap='viridis')
            # plt.title('Cross-Attention Matrix')
            # plt.xlabel('Legal Text Tokens')
            # plt.ylabel('Contract Text Tokens')
            # plt.show()
            # contract_attention_weights = attn_show.mean(axis=1)  # 平均每一列的权重
            # legal_attention_weights = attn_show.mean(axis=0)  # 平均每一行的权重
            #
            # highlighted_contract_text = self.highlight_text(contract_text, contract_attention_weights)
            # highlighted_legal_text = self.highlight_text(legal_text, legal_attention_weights)
            #

            row_sim = self.cosine_similarity(contracts, weiLaw, dim=2)

            if self.args.agg_func == 'LogSumExp':
                row_sim.mul_(self.args.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim) / self.args.lambda_lse
            elif self.args.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif self.args.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif self.args.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(self.args.agg_func))
            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)
        return similarities
class ContrastiveLoss(nn.Module):
    def __init__(self, args, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.margin = margin
        self.max_violation = max_violation
        self.xattn_score_con_law = ContractToLaw(self.args)

    def forward(self, con, law, law_len):

        scores = self.xattn_score_con_law(con, law, law_len)

        diagonal = scores.diag().view(con.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_law = (self.margin + scores - d1).clamp(min=0)
        cost_con = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = mask
        I = torch.eye(cost_law.size(0), cost_law.size(1)) > 0.5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_law = cost_law.masked_fill_(I, 0)
        cost_con = cost_con.masked_fill_(I, 0)

        if self.max_violation:
            cost_law = cost_law.max(1)[0]
            cost_con = cost_con.max(0)[0]
        return cost_law.sum() + cost_con.sum()
