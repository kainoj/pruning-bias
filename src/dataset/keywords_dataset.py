from typing import List

from torch.utils.data import Dataset
from src.utils.utils import get_logger


log = get_logger(__name__)


class SentencesWithKeywordsDataset(Dataset):
    """Returns tokenized sentence with a mask indicating position of keywords.

    Args:c
        sentences: list of sentences
        keywords: keywords for each sentence.
        tokenizer: 🤗

    Returns: a dict with the following keys:
        input_ids, attention_mask: standard 🤗's tokenizer output.
        keyword_mask: mask indicating on which position in tokenized
            sequenced are the attributes.
    """
    def __init__(
        self,
        sentences: List[str],
        keywords: List[set[str]],
        tokenizer
    ) -> None:
        super().__init__()

        self.sentences = sentences
        self.keywords = keywords
        self.tokenizer = tokenizer

        log.info(f"Total sentences with attributes: {len(self.sentences)}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        keywords = ' '.join(list(self.keywords[idx]))

        # Tokenize the sentence and make sure that every tensor is of shape
        # 'max_length' (so it batchifies properly)
        payload = self.tokenizer(sentence)
        payload = {key: val.squeeze(0) for key, val in payload.items()}

        # Tokens of the sentence
        sentence_tokens = payload['input_ids']

        # Remove CLS/SEP and reshape, so it broadcasts nicely
        keyword_tokens = self.tokenizer(keywords, padding=False)
        keyword_tokens = keyword_tokens['input_ids'][:, 1:-1].reshape((-1, 1))

        # Mask indicating positions of attributes within sentence econdings
        # Each row of (sent==attr) contains position of consecutive tokens
        mask = (sentence_tokens == keyword_tokens).sum(0)

        payload['keyword_mask'] = mask

        return payload
