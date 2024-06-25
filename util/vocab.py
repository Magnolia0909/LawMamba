# -*- coding: utf-8 -*-

import json
import jieba
import re
from cn2an import transform
import pandas as pd
from collections import Counter
all_texts = []

with open("/data/law_data/test_json_v3.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    for entry in data:
        if "contract" in entry:
            all_texts.append(entry["contract"])

with open("/data/law_data/train_json_v3.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
    for entry in test_data:
        if "contract" in entry:
            all_texts.append(entry["contract"])

law_df = pd.read_csv("/data/law_data/law.csv", encoding="utf-8")
all_texts.extend(law_df['law_content'])

vocab = {}
unk = "<UNK>"
pad = "<PAD>"
start = "<START>"
end = "<END>"
# num = "NUM"
next_index = 0
unique_words_set = set()
with open("/data/law_data/te/vocab.txt", "w", encoding="utf-8") as vocab_file:
    vocab_file.write(f"{pad} {next_index}\n")
    next_index += 1
    vocab_file.write(f"{unk} {next_index}\n")
    next_index += 1
    vocab_file.write(f"{start} {next_index}\n")
    next_index += 1
    vocab_file.write(f"{end} {next_index}\n")
    next_index += 1
    # vocab_file.write(f"{num},{next_index}\n")
    # next_index += 1
    # 统计词频，利用Counter可以直接按单个字符进行统计词频
    counter = Counter()
    for text in all_texts:
        clean_text = re.sub(r'\s+', '', text)
        clean_text = re.sub(r"[^\w\s,，。.《》]", "", clean_text)

        counter.update(clean_text)
        # # clean_text = transform(clean_text, 'an2cn')
        # word_tokens = list(jieba.cut(clean_text))
        #
        # for word in word_tokens:
        #     if word not in vocab:
        #         vocab[word] = next_index
        #         next_index += 1
        #         vocab_file.write(f"{word},{vocab[word]}\n")
        #
        # unique_words_set.update(word_tokens)
    tokens = [token for token, count in counter.items()]
    for token in tokens:
        if token not in vocab:
            vocab[token] = next_index
            next_index += 1
            vocab_file.write(f"{token} {vocab[token]}\n")
# 打开文件并计算行数
with open('/data/law_data/te/vocab.txt', "r", encoding="utf-8") as file:
    lines = file.readlines()
    line_count = len(lines)
# 输出行数
print(f"vocab 文件有 {line_count} 行.")
