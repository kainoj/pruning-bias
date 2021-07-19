from typing import List
import numpy as np
import scipy
import json

import torch
from torch import nn
from transformers import pipeline


def cosine_similarity(u: np.array, v: np.array) -> float:
    return 1.0 - scipy.spatial.distance.cosine(u, v)


def association(w, attribute_a, attribute_b):
    cos_a = np.array([cosine_similarity(w, a) for a in attribute_a])
    cos_b = np.array([cosine_similarity(w, b) for b in attribute_b])
    return cos_a.mean() - cos_b.mean()


def weat(
    target_x: np.array,
    target_y: np.array,
    attribute_a: np.array,
    attribute_b: np.array
) -> float:
    sum_x = sum([association(x, attribute_a, attribute_b) for x in target_x])
    sum_y = sum([association(y, attribute_a, attribute_b) for y in target_y])
    
    return sum_x - sum_y

def random_choice_2d(arr: np.array, size: int, replace: bool = False) -> np.array:
    """Randomly chooses subset of rows of given size."""
    idx = np.random.choice(arr.shape[0], size, replace)
    return arr[idx]

def weat_runner(
    target_x: np.array,
    target_y: np.array,
    attribute_a: np.array,
    attribute_b: np.array,
    num_reps: int = 1000
) -> float:

    subset_size_target = min(target_x.shape[0], target_y.shape[0]) // 2
    subset_size_attrib = min(attribute_a.shape[0], attribute_b.shape[0]) // 2

    scores = []

    print("Running WEAT...")

    for i in range(num_reps):
        s = weat(
            target_x=random_choice_2d(target_x, size=subset_size_target),
            target_y=random_choice_2d(target_y, size=subset_size_target),
            attribute_a=random_choice_2d(attribute_a, size=subset_size_attrib),
            attribute_b=random_choice_2d(attribute_b, size=subset_size_attrib)
        )
        scores.append(s)

        if i % 100 == 0:
            print(f'[i={i:0>3}] {np.mean(scores)}')

    return np.mean(scores)


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

    seat = weat_runner(
        target_x=x_repr, target_y=y_repr,
        attribute_a=a_repr, attribute_b=b_repr,
        num_reps=5000
    )


if __name__ == "__main__":
    main()
