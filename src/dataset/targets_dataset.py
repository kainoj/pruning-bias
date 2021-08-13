from typing import List

from torch.utils.data import Dataset
from src.utils.utils import get_logger

log = get_logger(__name__)


class SentencesWithTargetsDatset(Dataset):

    def __init__(
        self,
        sentences: List[str],
        targets_in_sentences: List[set[str]],
        tokenizer
    ) -> None:
        super().__init__()

        self.sentences = sentences
        self.targets_in_sentences = targets_in_sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # TODO: This is only for sentence-level debiasing
        #   For token-level, we need an additional mask for targets!

        y = self.tokenizer(sentence)

        # Make sure that every tensor is 1D of shape 'max_length'
        #  (so it batchifies properly)
        y = {key: val.squeeze(0) for key, val in y.items()}
        return y
