import torch

from seat import SEAT6, SEAT7, SEAT8


class SentenceEmbedder():

    def __init__(self, model, tokenizer, mode='CLS') -> None:

        self.model = model
        self.tokenizer = tokenizer

        if mode != 'CLS':
            raise NotImplementedError()

    def __call__(self, x) -> torch.tensor:

        inputs = self.tokenizer(x, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)

        # Take the embedding of CLS token as sentence representation
        return outputs.last_hidden_state[:, 0, :]


if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    model_name = 'distilbert-base-uncased'
    data_filename = 'data/sent-weat6.jsonl'

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embedder = SentenceEmbedder(model, tokenizer)

    seat_metrics = {"SEAT6": SEAT6(), "SEAT7": SEAT7(), "SEAT8": SEAT8()}

    for name in seat_metrics:
        value = seat_metrics[name](embedder)

        print(f"{name}: {value}")
