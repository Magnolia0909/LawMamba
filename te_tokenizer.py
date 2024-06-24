# -*- coding: utf-8 -*-
# @Time : 2023/9/1 下午3:11
# @Author : Xiuxuan Shen
import jieba
import jieba.analyse
import torch
import re
from cn2an import transform

from collections import Counter

class Tokenizer(object):
    def __init__(self, args):
        self.args = args
        self.max_seq_length = self.args.max_seq_length
        self.vocab = self.load_vocab()
        self.token2idx, self.idx2token = self.create_vocabulary()
        self.unknown_id = self.token2idx["<UNK>"]
        self.pad_id = self.token2idx["<PAD>"]
        self.bos_id = self.token2idx["<START>"]
        self.eos_id = self.token2idx["<END>"]

    def load_vocab(self):
        vocab = {}
        with open(self.args.te_vocab_file, "r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                word, token_id = line.strip().split(" ")
                vocab[word] = int(token_id)
            return vocab

    def replace_numbers(self, text: str, symbol: str) -> str:
        # 使用正则表达式匹配文本中的所有数字。
        pattern = r"\d+"
        matches = re.findall(pattern, text)

        # 用指定符号替换每个数字。
        for match in matches:
            text = text.replace(match, symbol, 1)  # 仅替换第一次出现的数字

        # 使用正则表达式匹配文本中的所有中文数字。
        pattern = r"[一二三四五六七八九十百千万亿]+"
        matches = re.findall(pattern, text)

        # 用指定符号括起来替换每个中文数字。
        for match in matches:
            text = text.replace(match, f"[{symbol}]", 1)  # 仅替换第一次出现的中文数字

        # 使用正则表达式匹配文本中所有连续的 [num]。
        pattern = rf"\[{symbol}]+"
        matches = re.findall(pattern, text)

        # 用单个 [num] 替换每个连续的 [num]。
        for match in matches:
            text = text.replace(match, f"[{symbol}]")

        return text

    def preprocess_sentence(self, sentence):
        num_pattern = r'num(?:num(?:num)?)?'
        num_replacement = 'NUM'

        with open(self.args.stopword_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f.readlines()]

        clean_text = transform(sentence, 'cn2an')
        clean_text = self.replace_numbers(clean_text, '[num]')
        clean_text = re.sub(r'\s+', '', clean_text)
        clean_text = re.sub(r"[^\w\s]", "", clean_text)
        clean_text = re.sub(num_pattern, num_replacement, clean_text)

        start_token = "<START>"
        end_token = "<END>"
        words = list(jieba.cut(clean_text))
        words.insert(0, start_token)
        words.append(end_token)
        words = [word for word in words if word not in stopwords]
        keywords = jieba.analyse.extract_tags(clean_text, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
        tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]

        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        attention_mask = [1] * len(tokens) + [0] * (self.max_seq_length - len(tokens))
        token_type_ids = [0] * self.max_seq_length

        if len(tokens) < self.max_seq_length:
            tokens.extend([self.vocab["<PAD>"]] * (self.max_seq_length - len(tokens)))

        tokens = torch.tensor(tokens).unsqueeze(0)
        attention_masks = torch.tensor(attention_mask).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)
        return tokens, attention_masks, token_type_ids
    def encode(self, sentence):
        tokens = [self.bos_id, ]  # 起始标记
        # 遍历，词转编号
        for token in sentence:
            tokens.append(self.get_id_by_token(token))
        tokens.append(self.eos_id)  # 结束标记

        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        attention_mask = [1] * len(tokens) + [0] * (self.max_seq_length - len(tokens))
        token_type_ids = [0] * self.max_seq_length

        if len(tokens) < self.max_seq_length:
            tokens.extend([self.vocab["<PAD>"]] * (self.max_seq_length - len(tokens)))

        tokens = torch.tensor(tokens).unsqueeze(0)
        attention_masks = torch.tensor(attention_mask).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)
        return tokens, attention_masks, token_type_ids

    def create_vocabulary(self):
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(self.vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        return token2idx, idx2token

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<UNK>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode_seq(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx != self.bos_id and idx != self.eos_id:
                txt += self.idx2token[int(idx)]
            else:
                break
        return txt

    def decode_token(self, ids):
        out = []
        for i, idx in enumerate(ids):
            if idx != self.bos_id and idx != self.eos_id:
                out.append(self.idx2token[int(idx)])
            else:
                break
        return out

    def decode_dp(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch, mode=None):
        out = []
        if mode == "seq":
            for ids in ids_batch:
                out.append(self.decode_seq(ids))
        elif mode == "token":
            for ids in ids_batch:
                out.append(self.decode_token(ids))
        elif mode == "dp":
            for ids in ids_batch:
                out.append(self.decode_dp(ids))
        return out
