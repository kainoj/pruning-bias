
import json

from typing import Callable, List, Optional, Tuple
from torch.utils.data import Dataset


class WeatDataset(Dataset):

    def __init__(
        self,
        data_filename: str,
        tokenizer: Optional[Callable] = None
    ) -> None:
        """Dataset for serving targets/attributes samples for the WEAT test.

        Args:
            data_filename: path to .json(l) file containing data for the WEAT
                test. The file must contain the following keys: `targ1`,
                `targ2`, `attr1`, `attr2`.
            tokenizer: anything that takes a string and returns tokens or
                embeddings. If `None`, return a plain sting
        """
        self.data_filename = data_filename
        self.tokenizer = tokenizer

        self.target_x, self.target_y, self.attribute_a, self.attribute_b = \
            self._get_data()

    def __len__(self):
        return min(
            len(self.target_x),
            len(self.target_y),
            len(self.attribute_a),
            len(self.attribute_b)
        )

    def __getitem__(self, idx):
        if self.tokenizer:
            return (
                self.tokenizer(self.target_x[idx]),
                self.tokenizer(self.target_y[idx]),
                self.tokenizer(self.attribute_a[idx]),
                self.tokenizer(self.attribute_b[idx])
            )

        return (
            self.target_x[idx],
            self.target_y[idx],
            self.attribute_a[idx],
            self.attribute_b[idx]
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
