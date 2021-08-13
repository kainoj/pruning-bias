import torch
import torch.nn as nn
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
        attribute_b: torch.tensor
    ) -> None:
        self.target_x.append(target_x)
        self.target_y.append(target_y)
        self.attribute_a.append(attribute_a)
        self.attribute_b.append(attribute_b)

    def compute(self) -> float:
        """Computes WEAT effect size.

        Returns: effect size of WEAT.
        """
        return self.effect_size(
            torch.vstack(self.target_x),
            torch.vstack(self.target_y),
            torch.vstack(self.attribute_a),
            torch.vstack(self.attribute_b)
        )

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
