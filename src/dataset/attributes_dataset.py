from typing import Any, Dict, List
from collections import defaultdict 

import regex as re


from pathlib import Path
from torch.utils.data import Dataset
from src.utils.utils import get_logger


log = get_logger(__name__)


class AttributesDataset(Dataset):

    def __init__(self, sentences: List[str], attr2sents: Dict[str, List[str]]) -> None:
        super().__init__()
        self.sentences = sentences
        self.attr2sents = attr2sents

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence, attributes = self.sentences[idx]
        y = [self.attr2sents[a] for a in attributes]
        # TODO
        return sentence


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

    # Get lists of attributes
    male_attr = get_attribute_set(male_attr_path)
    female_attr = get_attribute_set(female_attr_path)
    stereo_attr = get_attribute_set(stereo_attr_path)

    # This regexp basically tokenizes a sentence over spaces and 's, 're, 've..
    # It's originally taken from OpenAI's GPT-2 Encoder implementation
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # Each sentences of the list contains at least one of M/F/S attribute
    male_sents, female_sents, stereo_sents = [], [], []

    # i-th element tells us which attributes are in the i-th sentence
    male_sents_attr, female_sents_attr, stereo_sents_attr = [], [], []

    # Dictionary mapping attributes to sentences containing that attributes
    attr2sents = defaultdict(list)

    with open(rawdata_path) as f:
        for full_line in f.readlines():

            line = full_line.strip()

            # This is how they decied to filter out data in the paper
            if len(line) < 1 or len(line.split()) > 128 or len(line.split()) <= 1:
                continue

            line_tokenized = {token.strip().lower() for token in re.findall(pat, line)}
            
            # Dicts containing M/F/S attributes in each sentence
            male = line_tokenized & male_attr
            female = line_tokenized & female_attr
            stereo = line_tokenized & stereo_attr

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
                    attr2sents[f].append(female)
                
            # Sentences with stereotype attributes
            if len(stereo) > 0 and len(male) == 0 and len(female) == 0: 
                stereo_sents.append(line)
                stereo_sents_attr.append(stereo)

                for s in stereo:
                    attr2sents[s].append(line)

    return {
        'male_sents': male_sents, 'male_sents_attr': male_sents_attr,
        'female_sents': female_sents, 'female_sents_attr': female_sents_attr,
        'stereo_sents': stereo_sents, 'stereo_sents_attr': stereo_sents_attr,
        'attributes': attr2sents
    }
