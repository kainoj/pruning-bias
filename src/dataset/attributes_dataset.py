from typing import Any, Dict, List

import torch

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
        self.attributes = attributes
        self._tokenizer = tokenizer  # tokenizer should be accessed via tokenize()

        """
        For targets we need to keep:
            - List[ (sentences, List[keywords] )]
        For attributes we need to keep:
            - List[ (List[sentences], keywords)]

        In either cases, in getter, we return
            - tokenized sentence with keyword mask
        """

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

        payload = self.tokenize(sentence)
        
        # Make sure that every tensor is 1D of shape 'max_length' (so it batchifies properly)
        payload = {key: val.squeeze(0) for key, val in payload.items()}

        # Tokens of the sentence
        sent = payload['input_ids']

        # Tokens of attribute (might be more than 1). Remove CLS/SEP and reshape, so it broadcasts nicely
        attr = self.tokenize(attr, padding=False)['input_ids'][1:-1].reshape((-1, 1))

        # Mask indicating positions of attributes within sentence econdings
        # Each row of (sent==attr) contains position of consecutive tokens
        mask = (sent == attr).sum(0)

        payload['attribute_mask'] = mask
        payload['attribute_id'] = attr_id
        
        return payload

    def tokenize(self, sentence, padding='max_length'):
        """Wrapper for tokenizer to ensure every sentence gets padded to same length"""
        return self._tokenizer(
            sentence,
            padding=padding,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
  
