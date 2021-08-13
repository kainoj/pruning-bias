from src.metrics.weat import WEAT


class SEAT(WEAT):
    """Sentence Embedding Association Test (SEAT).

    SEAT is just a WEAT that takes representation of the whole sentence、for
    example an embedding of [CLS] token.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
