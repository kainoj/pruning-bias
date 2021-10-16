from typing import List

from torch.utils.data import Dataset
from src.utils.utils import get_logger


log = get_logger(__name__)


class AttributesWithSentencesDataset(Dataset):
    """Dataset maintaining sentences and their attributes.

    Args:
        sentences: list of sentences
        attributes: attributes for each sentence
        tokenizer: 🤗

    Returns: a dict with the following keys:
        input_ids, attention_mask: standard 🤗's tokenizer output.
        attribute_mask: mask indicating on which position in tokenized
            sequenced are the attributes.
        attribute_gender: gender of an attribute 0/1
    """
    def __init__(
        self,
        sentences: List[str],
        attributes: List[set[str]],
        tokenizer
    ) -> None:
        super().__init__()

        self.sentences = sentences
        self.attributes = attributes
        self.tokenizer = tokenizer

        log.info(f"Total sentences with attributes: {len(self.sentences)}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        attributes = ' '.join(list(self.attributes[idx]))

        # Tokenize the sentence and make sure that every tensor is of shape
        # 'max_length' (so it batchifies properly)
        payload = self.tokenizer(sentence)
        payload = {key: val.squeeze(0) for key, val in payload.items()}

        # Tokens of the sentence
        sentence_tokens = payload['input_ids']

        # Remove CLS/SEP and reshape, so it broadcasts nicely
        attribute_tokens = self.tokenizer(attributes, padding=False)
        attribute_tokens = attribute_tokens['input_ids'][:, 1:-1].reshape((-1, 1))

        # Mask indicating positions of attributes within sentence econdings
        # Each row of (sent==attr) contains position of consecutive tokens
        mask = (sentence_tokens == attribute_tokens).sum(0)

        payload['attribute_mask'] = mask

        return payload
