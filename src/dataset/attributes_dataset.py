from typing import Any, Dict, List
from collections import defaultdict 
from functools import partial
from pathlib import Path

import regex as re
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
  
def get_attribute_set(filepath: str) -> set:
    """Reads file with attributes and returns a set containing them all"""

    # This is cumbersome: hydra creates own build dir,
    # where data/ is not present. We need to escape to original cwd
    import hydra  # TODO(Przemek): do it better. Maybe download data?
    quickfix_path = Path(hydra.utils.get_original_cwd()) / filepath

    with open(quickfix_path) as f:
        return {l.strip() for l in f.readlines()}


def extract_data(
    rawdata_path: Path,
    male_attr_path: Path,
    female_attr_path: Path,
    stereo_attr_path: Path
) -> Any: # TODO type
    # TODO: refactor names so it's clear what's attribute and what's target
    # Get lists of attributes
    male_attr = get_attribute_set(male_attr_path)
    female_attr = get_attribute_set(female_attr_path)
    stereo_trgt = get_attribute_set(stereo_attr_path)

    # This regexp basically tokenizes a sentence over spaces and 's, 're, 've..
    # It's originally taken from OpenAI's GPT-2 Encoder implementation
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # Each sentences of the list contains at least one of M/F/S attribute
    male_sents, female_sents, stereo_sents = [], [], []

    # i-th element tells us which attributes/targets are in the i-th sentence
    male_sents_attr, female_sents_attr, stereo_sents_trgt = [], [], []

    # Dictionary mapping attributes to sentences containing that attributes
    attr2sents = defaultdict(list)

    with open(rawdata_path) as f:
        for full_line in f.readlines():

            line = full_line.strip()

            # This is how they decied to filter out data in the paper
            if len(line) < 1 or len(line.split()) > 128 or len(line.split()) <= 1:
                continue

            line_tokenized = {token.strip().lower() for token in re.findall(pat, line)}
            
            # Dicts containing M/F/S attributes/targets in each sentence
            male = line_tokenized & male_attr
            female = line_tokenized & female_attr
            stereo = line_tokenized & stereo_trgt

            # Note that a sentence might contain attributes of only one category
            #   M/F/S. That's why we check for emptiness of other sets.

            # Sentences with male attributes
            if len(male) > 0 and len(female) == 0:
                male_sents.append(line)
                male_sents_attr.append(male)

                for m in male:
                    attr2sents[m].append(line)

            # Sentences with female attributes
            if len(female) > 0 and len(male) == 0:
                female_sents.append(line)
                female_sents_attr.append(female)

                for f in female:
                    attr2sents[f].append(line)
                
            # Sentences with stereotype target
            if len(stereo) > 0 and len(male) == 0 and len(female) == 0: 
                stereo_sents.append(line)
                stereo_sents_trgt.append(stereo)

                for s in stereo:
                    attr2sents[s].append(line)

    return {
        'male_sents': male_sents, 'male_sents_attr': male_sents_attr,
        'female_sents': female_sents, 'female_sents_attr': female_sents_attr,
        'stereo_sents': stereo_sents, 'stereo_sents_trgt': stereo_sents_trgt,
        'attributes': attr2sents
    }
