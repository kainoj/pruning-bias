from typing import Tuple

import torch
import torch.nn as nn

from torch import Tensor
from torchmetrics import Metric


class WEAT(Metric):
    """Word Embedding Association Test (WEAT).

    Based on:
        Caliskan et al, 2017, "Semantics derived automatically from language
        corpora contain human-like biases"

    Ref.: https://www.cs.bath.ac.uk/~jjb/ftp/CaliskanEtAl-authors-full.pdf
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The following should store *embeddigs* of targets and attributes.
        self.add_state("target_x", default=[], dist_reduce_fx="cat")
        self.add_state("target_y", default=[], dist_reduce_fx="cat")
        self.add_state("attribute_a", default=[], dist_reduce_fx="cat")
        self.add_state("attribute_b", default=[], dist_reduce_fx="cat")

    def update(
        self,
        target_x: torch.tensor,
        target_y: torch.tensor,
        attribute_a: torch.tensor,
        attribute_b: torch.tensor,
    ) -> None:
        self.target_x.append(target_x)
        self.target_y.append(target_y)
        self.attribute_a.append(attribute_a)
        self.attribute_b.append(attribute_b)

    def compute(self) -> float:
        """Computes WEAT effect size.

        Returns: effect size of WEAT.
        """
        x, y, a, b = self._get_final_stats()
        return self.effect_size(target_x=x, target_y=y, attribute_a=a, attribute_b=b)

    def _get_final_stats(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Concats score items if necessary, before passing them to a compute function."""

        def _convert(x) -> Tensor:
            return torch.vstack(x) if isinstance(x, list) else x

        x = _convert(self.target_x)
        y = _convert(self.target_y)
        a = _convert(self.attribute_a)
        b = _convert(self.attribute_b)
        return x, y, a, b

    def s_wAB(
        self,
        w: torch.tensor,
        attribute_a: torch.tensor,
        attribute_b: torch.tensor
    ) -> float:
        """Differential association of a word `w` with sets of attributes."""

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
