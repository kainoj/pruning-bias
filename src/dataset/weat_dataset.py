import json

from typing import Callable, List, Tuple
import torch
from torch.utils.data import Dataset


class WeatDataset(Dataset):

    def __init__(
        self,
        data_filename: str,
        tokenizer: Callable[[str], torch.tensor]
    ) -> None:
        """Dataset for serving targets/attributes samples for the WEAT test.

        Args:
            data_filename: path to .json(l) file containing data for the WEAT
                test. The file must contain the following keys: `targ1`,
                `targ2`, `attr1`, `attr2`.
            tokenizer: anything that takes a string and returns tokens.
        """
        self.data_filename = data_filename
        self.tokenizer = tokenizer

        self.target_x, self.target_y, self.attribute_a, self.attribute_b = \
            self._get_data()

    def __len__(self):
        return max(
            len(self.target_x),
            len(self.target_y),
            len(self.attribute_a),
            len(self.attribute_b)
        )

    def tokenize_and_squeeze(self, sentence: str):
        """Tokenizes a sentence and squeezes tensors (so it batchifies properly.
        """
        y = self.tokenizer(sentence)
        return {key: val.squeeze(0) for key, val in y.items()}

    def get_single_item(self, sentences: List[str], idx: int):
        """Get `idx`-th of `sentences`.

        If index exceeds the list, some random item is returned and along
        with `is_legit` flag telling us whether the item is should be considered
        in later WEAT score calculations.

        This is because we want all items from all target/attribute lists,
        but these lists doesn't need to be of equal length.
        """
        length = len(sentences)
        x = sentences[idx % length]
        is_legit = (idx < length)

        y = self.tokenizer(x)
        y['is_legit'] = torch.tensor(is_legit)

        return {key: val.squeeze(0) for key, val in y.items()}

    def get_all_items(self):
        return {
            'target_x': self.tokenizer(self.target_x, return_tensors='pt'),
            'target_y': self.tokenizer(self.target_y, return_tensors='pt'),
            'attribute_a': self.tokenizer(self.attribute_a, return_tensors='pt'),
            'attribute_b': self.tokenizer(self.attribute_b, return_tensors='pt'),
        }

    def __getitem__(self, idx):
        return (
            self.get_single_item(self.target_x, idx),
            self.get_single_item(self.target_y, idx),
            self.get_single_item(self.attribute_a, idx),
            self.get_single_item(self.attribute_b, idx)
        )

    def _get_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Load data for the WEAT test

        Retruns: two lists of targets and two lists of attributes.
        """
        with open(self.data_filename) as f:
            data = json.load(f)

        target_x = data['targ1']['examples']
        target_y = data['targ2']['examples']
        attribute_a = data['attr1']['examples']
        attribute_b = data['attr2']['examples']

        return target_x, target_y, attribute_a, attribute_b
