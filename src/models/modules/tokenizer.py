from transformers import AutoTokenizer


class Tokenizer():
    """An unified wrapper for the tokenier.

    Just to make sure that every sentence gets padded to same length.
    """

    def __init__(self, model_name, max_length=128) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, sentence: str, padding='max_length', return_tensors=None):
        """Tokenizer call.

        Use:
            - return_tensors=None for 🤗datasets
            - return_tensors='pt' for torch-style dataset
        """
        return self._tokenizer(
            sentence,
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )

    def decode(self, x):
        return self._tokenizer.decode(x)
