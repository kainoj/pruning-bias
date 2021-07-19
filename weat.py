import json

import torch
import torch.nn as nn
from typing import Any, Callable  # Tuple, List


class WEAT():

    def __init__(self, data_filename: str) -> None:
        """TODO
        """
        self.target_x, self.target_y, self.attribute_a, self.attribute_b = \
            self._get_data(data_filename)

    def _get_data(self, data_filename: str):
        """ASd"""
        with open(data_filename) as f:
            data = json.load(f)

        target_x = data['targ1']['examples']
        target_y = data['targ2']['examples']
        attribute_a = data['attr1']['examples']
        attribute_b = data['attr2']['examples']

        return target_x, target_y, attribute_a, attribute_b

    def association(
        self, w: torch.tensor, attribute_a: torch.tensor, attribute_b: torch.tensor
    ) -> float:
        """Association of a word with an attribute. Torch version.
        """
        _w = w.unsqueeze(0)
        assoc_a = nn.CosineSimilarity(dim=1)(_w, attribute_a)
        assoc_b = nn.CosineSimilarity(dim=1)(_w, attribute_b)
        return assoc_a.mean() - assoc_b.mean()

    def weat(self, target_x, target_y, attribute_a, attribute_b) -> float:
        sum_x = sum([self.association(x, attribute_a, attribute_b) for x in target_x])
        sum_y = sum([self.association(y, attribute_a, attribute_b) for y in target_y])
        return sum_x - sum_y

    def weat_effect_size(
        self,
        target_x: torch.tensor,
        target_y: torch.tensor,
        attribute_a: torch.tensor,
        attribute_b: torch.tensor,
    ) -> float:

        target_xy = torch.vstack((target_x, target_y))

        assoc_x = [self.association(x, attribute_a, attribute_b) for x in target_x]
        assoc_y = [self.association(y, attribute_a, attribute_b) for y in target_y]
        assoc_xy = [self.association(xy, attribute_a, attribute_b) for xy in target_xy]

        assoc_x = torch.tensor(assoc_x)
        assoc_y = torch.tensor(assoc_y)
        assoc_xy = torch.tensor(assoc_xy)

        return (assoc_x.mean() - assoc_y.mean()) / assoc_xy.std()

    def __call__(self, embedder: Callable, *args: Any, **kwds: Any) -> float:

        x_emb = embedder(self.target_x)
        y_emb = embedder(self.target_y)
        a_emb = embedder(self.attribute_a)
        b_emb = embedder(self.attribute_b)

        return self.weat_effect_size(x_emb, y_emb, a_emb, b_emb)
