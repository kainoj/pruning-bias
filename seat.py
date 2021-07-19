from weat import WEAT


class SEAT6(WEAT):

    def __init__(self) -> None:
        super().__init__(data_filename='data/sent-weat6.jsonl')


class SEAT7(WEAT):

    def __init__(self) -> None:
        super().__init__(data_filename='data/sent-weat7.jsonl')


class SEAT8(WEAT):

    def __init__(self) -> None:
        super().__init__(data_filename='data/sent-weat8.jsonl')
