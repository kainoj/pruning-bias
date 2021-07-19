from typing import List
import numpy as np
import scipy
import json

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from weat import get_data


def association(w, attribute_a: torch.tensor, attribute_b: torch.tensor) -> float:
    """Association of a word with an attribute. Torch version.
    """
    _w = w.unsqueeze(0)
    assoc_a = nn.CosineSimilarity(dim=1)(_w, attribute_a)
    assoc_b = nn.CosineSimilarity(dim=1)(_w, attribute_b)
    return assoc_a.mean() - assoc_b.mean()

def weat(target_x, target_y, attribute_a, attribute_b) -> float:
    """
    """    
    sum_x = sum([association(x, attribute_a, attribute_b) for x in target_x])
    sum_y = sum([association(y, attribute_a, attribute_b) for y in target_y])
    return sum_x - sum_y

def weat_effect_size(
    target_x: torch.tensor,
    target_y: torch.tensor,
    attribute_a: torch.tensor,
    attribute_b: torch.tensor,
) -> float:

    target_xy = torch.vstack((target_x, target_y))

    assoc_x = torch.tensor([association(x, attribute_a, attribute_b) for x in target_x])
    assoc_y = torch.tensor([association(y, attribute_a, attribute_b) for y in target_y])
    assoc_xy = torch.tensor([association(xy, attribute_a, attribute_b) for xy in target_xy])

    return (assoc_x.mean() - assoc_y.mean()) / assoc_xy.std()


def get_representation(sentences: List[str], tokenizer, model) -> torch.tensor:
    
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    # outputs shape is (#sentences, #tokens, dim)
    outputs = model(**inputs).last_hidden_state

    # Take the embedding of CLS token as sentence representation
    return outputs[:, 0, :]

def main():
    model_name = 'distilbert-base-uncased'
    data_filename = 'data/sent-weat6.jsonl'

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    target_x, target_y, attribute_a, attribute_b = get_data(data_filename)

    x_repr, y_repr, a_repr, b_repr = [
        get_representation(dat, tokenizer, model) for dat in [target_x, target_y, attribute_a, attribute_b]
    ]

    print(
        [arr.shape for arr in [x_repr, y_repr, a_repr, b_repr]]
    )

    y = weat_effect_size(
        target_x=x_repr, target_y=y_repr,
        attribute_a=a_repr, attribute_b=b_repr
    )
    print(f"Effect size: {y}")


if __name__ == "__main__":
    main()
