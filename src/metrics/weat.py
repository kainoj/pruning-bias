import json

import torch
import torch.nn as nn
from typing import Callable, List, Tuple


class WEAT():
    """Word Embedding Association Test (WEAT).

    Based on:
        Caliskan et al, 2017, "Semantics derived automatically from language
        corpora contain human-like biases"

    Ref.: https://www.cs.bath.ac.uk/~jjb/ftp/CaliskanEtAl-authors-full.pdf
    """

    def __init__(self, data_filename: str) -> None:
        """WEAT

        Args:
            data_filename: path to .jsonl file containing test data.
        """
        self.target_x, self.target_y, self.attribute_a, self.attribute_b = \
            self._get_data(data_filename)

    def _get_data(
        self,
        data_filename: str
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Load data for the WEAT test

        Args:
            data_filename: path to .jsonl file containing test data.

        Retruns: two lists of targets and two lists of attributes.
        """
        with open(data_filename) as f:
            data = json.load(f)

        target_x = data['targ1']['examples']
        target_y = data['targ2']['examples']
        attribute_a = data['attr1']['examples']
        attribute_b = data['attr2']['examples']

        return target_x, target_y, attribute_a, attribute_b

    def s_wAB(
        self,
        w: torch.tensor,
        attribute_a: torch.tensor,
        attribute_b: torch.tensor
    ) -> float:
        """Differencial association of a word `w` with sets of attributes."""

        _w = w.unsqueeze(0)
        assoc_a = nn.CosineSimilarity(dim=1)(_w, attribute_a)
        assoc_b = nn.CosineSimilarity(dim=1)(_w, attribute_b)
        return assoc_a.mean() - assoc_b.mean()

    def s_XYAB(
        self,
        target_x: torch.tensor,
        target_y: torch.tensor,
        attribute_a: torch.tensor,
        attribute_b: torch.tensor
    ) -> float:
        """Differential association of 2 sets of target words with attributes"""

        sum_x = sum([self.s_wAB(x, attribute_a, attribute_b) for x in target_x])
        sum_y = sum([self.s_wAB(y, attribute_a, attribute_b) for y in target_y])
        return sum_x - sum_y

    def effect_size(
        self,
        target_x: torch.tensor,
        target_y: torch.tensor,
        attribute_a: torch.tensor,
        attribute_b: torch.tensor
    ) -> float:
        """Measures the effect size.

        Effect size is "is a normalized measure of how separated the two
        distributions (of associations between the target and attribute) are".
        """

        target_xy = torch.vstack((target_x, target_y))

        assoc_x = [self.s_wAB(x, attribute_a, attribute_b) for x in target_x]
        assoc_y = [self.s_wAB(y, attribute_a, attribute_b) for y in target_y]
        assoc_xy = [self.s_wAB(xy, attribute_a, attribute_b) for xy in target_xy]

        assoc_x = torch.tensor(assoc_x)
        assoc_y = torch.tensor(assoc_y)
        assoc_xy = torch.tensor(assoc_xy)

        return (assoc_x.mean() - assoc_y.mean()) / assoc_xy.std()

    def __call__(self, embedder: Callable) -> float:
        """WEAT effect size.

        Args:
            embedder: anything that takes a list of sentences (strings)
                and retruns their embeddigs

        Returns: effect size of WEAT.
        """
        x_emb = embedder(self.target_x)
        y_emb = embedder(self.target_y)
        a_emb = embedder(self.attribute_a)
        b_emb = embedder(self.attribute_b)

        return self.effect_size(x_emb, y_emb, a_emb, b_emb)
