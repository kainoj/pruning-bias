from typing import List

from torch.utils.data import Dataset
from src.utils.utils import get_logger


log = get_logger(__name__)


class AttributesWithSentecesDataset(Dataset):

    def __init__(
        self,
        attributes: List[str],
        sentences_of_attributes: List[List[str]],
        tokenizer
    ) -> None:
        super().__init__()

        log.warning("ATTRIBUTES ARE TRIMMEND FOR FASTER DEVELOPMENT")
        """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
        attributes = attributes[:5]
        sentences_of_attributes = sentences_of_attributes[:5]
        cntr = 0
        for i, a in enumerate(attributes):
            log.info(f'{a} \t: {len(sentences_of_attributes[i])} sentences')
            cntr += len(sentences_of_attributes[i])
        log.info(f'    {cntr} sentences in total')
        """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

        self.attributes = attributes
        self.tokenizer = tokenizer

        self.sentences: List[tuple[int, str]] = []

        # Unroll sentences into one list, self.sentences[i] contains
        #   a sentence and its reference to its original atttribute
        for attr_id in range(len(self.attributes)):
            for sent in sentences_of_attributes[attr_id]:
                self.sentences.append((attr_id, sent))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        attr_id, sentence = self.sentences[idx]
        attr = self.attributes[attr_id]

        payload = self.tokenizer(sentence)

        # Make sure that every tensor is 1D of shape 'max_length' (so it batchifies properly)
        payload = {key: val.squeeze(0) for key, val in payload.items()}

        # Tokens of the sentence
        sent = payload['input_ids']

        # Tokens of attribute (might be more than 1).
        # Remove CLS/SEP and reshape, so it broadcasts nicely
        attr = self.tokenizer(attr, padding=False)['input_ids'][:, 1:-1].reshape((-1, 1))

        # Mask indicating positions of attributes within sentence econdings
        # Each row of (sent==attr) contains position of consecutive tokens
        mask = (sent == attr).sum(0)

        payload['attribute_mask'] = mask
        payload['attribute_id'] = attr_id

        return payload
