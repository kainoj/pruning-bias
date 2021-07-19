from typing import List
from transformers import AutoModel, AutoTokenizer
import torch

class SentenceEmbedder():

    def __init__(self, model_name: str, mode: str='CLS') -> None:
        """Wrapper for 🤗's pipeline abstraction with sentence representation.

        Args:
            model_name: e.g.: `distilbert-base-uncased`
            mode: how to represent sentence given tokens embeddings? 
                'CLS': sentence representation as embedding of [CLS] token
                 (taken at the last hidden state)
        """

        if mode != 'CLS':
            raise NotImplementedError()
        
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def __call__(self, sentences: List[str]) -> torch.tensor:

        # Tokenizer does automatically prepend [CLS] to sentences.
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)

        # Take the embedding of CLS token as sentence representation
        return outputs.last_hidden_state[:, 0, :]


if __name__ == "__main__":
    from seat import SEAT6, SEAT7, SEAT8

    model_name = 'distilbert-base-uncased'
    
    embedder = SentenceEmbedder(model_name)

    seat_metrics = {"SEAT6": SEAT6(), "SEAT7": SEAT7(), "SEAT8": SEAT8()}

    for name in seat_metrics:
        value = seat_metrics[name](embedder)
        print(f"{name}: {value}")
