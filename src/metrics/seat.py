from src.metrics.weat import WEAT
from pathlib import Path
import hydra

"""These are SEAT score wrappers, as proposed by May et al, 2019:
"On Measuring Social Biases in Sentence Encoders"

Ref. https://arxiv.org/abs/1903.10561

SEAT is just a WEAT that takes representation of the whole sentence.
"""

cwd = Path(hydra.utils.get_original_cwd())

class SEAT6(WEAT):

    def __init__(self) -> None:
        super().__init__(data_filename=cwd / 'data/sent-weat6.jsonl')


class SEAT7(WEAT):

    def __init__(self) -> None:
        super().__init__(data_filename=cwd / 'data/sent-weat7.jsonl')


class SEAT8(WEAT):

    def __init__(self) -> None:
        super().__init__(data_filename=cwd / 'data/sent-weat8.jsonl')
