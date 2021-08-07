from typing import Any, Dict, List

import torch

from torch.utils.data import Dataset
from src.utils.utils import get_logger


log = get_logger(__name__)


class AttributesDataset(Dataset):

    def __init__(
        self,
        sentences: List[str],
        targets_in_sentences: List[set[str]],
        attr2sent: dict[str, List[str]],
        tokenizer
    ) -> None:     
        super().__init__()
        self.sentences = sentences
        self.targets_in_sentences = targets_in_sentences
        self.attr2sent = attr2sent
        self._tokenizer = tokenizer  # tokenizer should be accessed via tokenize()

        """
        For targets we need to keep:
            - List[ (sentences, List[keywords] )]
        For attributes we need to keep:
            - List[ (List[sentences], keywords)]

        In either cases, in getter, we return
            - tokenized sentence with keyword mask
        """

    def __len__(self):
        return len(self.sentences)

    def get_attributes_with_sentences(self) -> List[dict[str, torch.tensor]]:
        """
        For each attribute:
            We want to have dict with fields:
                - inputs_ids - tokenized sentences
                - attention_mask - for tokenized sentences
                - attribute_indices - indices of tokenized attributes in the sentence
        This function would be called only once at epoch

        TODO: we can actually cache these results
        """
        res = []
    
        for attr, sents in self.attr2sent.items():
            # Tokenize the attribute and remove CLS/SEP
            attr_tokenized = self.tokenize(attr, padding=False)['input_ids'][:, 1:-1]
    
            # Tokenize all sentences connected with the attribute
            sents_tokenized = self.tokenize(sents)

            # Build a boolean mask such that 
            # mask[i, j]==True iff sents_tokenized[i, j] is a token of the attribute
            mask  = torch.full_like(sents_tokenized['input_ids'], fill_value=False)

            # Build that mask token by token
            for a in attr_tokenized.flatten():
                mask = mask | (a == sents_tokenized['input_ids'])

            # Add that mask to the input dict
            sents_tokenized['attributes_mask'] = mask

            res.append(sents_tokenized)

        return res

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # TODO: This is only for sentence-level debiasing
        #   For token-level, we need an additional mask for targets!
        print(sentence)

        y = self.tokenize(sentence)

        # Make sure that every tensor 1D of shape 'max_length'
        #  (so it batchifies properly)
        y = {key: val.squeeze(0) for key, val in y.items()}
        return y

    def tokenize(self, sentence, padding='max_length'):
        """Wrapper for tokenizer to ensure every sentence gets padded to same length"""
        return self._tokenizer(
            sentence,
            padding=padding,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
  
