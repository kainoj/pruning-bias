from typing import List
import numpy as np
import json

import torch
from torch import nn
from transformers import pipeline


def get_data(file_name):
    with open(file_name) as f:
        data = json.load(f)

    target_x = data['targ1']['examples']
    target_y = data['targ2']['examples']
    attribute_a = data['attr1']['examples']
    attribute_b = data['attr2']['examples']

    print(f"|X| = {len(target_x)}   \t |Y| = {len(target_y)}\n" \
          f"|A| = {len(attribute_a)}\t |B| = {len(attribute_b)}"
    )

    return target_x, target_y, attribute_a, attribute_b

def get_embeddings(pipeline, x, y, a, b):
    x_emb = pipeline(x)
    y_emb = pipeline(y)
    a_emb = pipeline(a)
    b_emb = pipeline(b)
    return (np.array(arr) for arr in [x_emb, y_emb, a_emb, b_emb])

def get_repr(x, y, a, b):
    """Get representation for each sentence, i.e. embedding of [CLS] token

    CLS token in the first token in each sequence.
    """
    return x[:, 0, :], y[:, 0, :], a[:, 0, :], b[:, 0, :]


def main():
    model_name = 'distilbert-base-uncased'
    data_file_name = 'data/sent-weat7b.jsonl'

    feature_extractor = pipeline(
        'feature-extraction',
        model=model_name,
        tokenizer=model_name,
        framework='pt'
    )
    
    target_x, target_y, attribute_a, attribute_b = get_data(data_file_name)

    x_emb, y_emb, a_emb, b_emb = get_embeddings(
        feature_extractor, target_x, target_y, attribute_a, attribute_b
    )

    x_repr, y_repr, a_repr, b_repr = get_repr(x_emb, y_emb, a_emb, b_emb)

    print(
        [arr.shape for arr in [x_repr, y_repr, a_repr, b_repr]]
    )


if __name__ == "__main__":
    main()
