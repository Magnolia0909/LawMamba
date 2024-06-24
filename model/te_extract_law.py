# -*- coding: utf-8 -*-
# @Time : 2023/10/30 上午11:32
# @Author : Xiuxuan Shen

import pandas as pd
from .te_tokenizer import Tokenizer
# from transformers import BertTokenizer
def extract_law(args):
    tokenizer = Tokenizer(args)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    vocab = tokenizer.load_vocab()
    law_df = pd.read_csv(args.law_data)
    law_input_ids = []
    law_attention_masks = []
    law_token_type_ids = []

    for index, row in law_df.iterrows():
        law_content = row['law_content']
        law_input_id, law_attention_mask, law_token_type_id = (tokenizer.encode(law_content))
        # law_input_id, law_attention_mask, law_token_type_id = Tokenizer(law_content, return_tensors='pt',
        #                                                                 max_length=args.max_seq_length,
        #                                                                 padding='max_length', truncation=True)

        law_input_ids.append(law_input_id)
        law_attention_masks.append(law_attention_mask)
        law_token_type_ids.append(law_token_type_id)

    law_df['law_input_ids'] = law_input_ids
    law_df['law_attention_mask'] = law_attention_masks
    law_df['law_token_type_ids'] = law_token_type_ids

    return law_df