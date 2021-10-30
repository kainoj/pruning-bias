from typing import Dict, List
from pathlib import Path
from datasets import load_dataset
from src.models.modules.tokenizer import Tokenizer
from datasets.utils import disable_progress_bar

import regex as re
import torch

disable_progress_bar()


def get_keyword_set(filepath: str) -> set:
    """Reads file with keywords and returns a set containing them all"""

    # This is cumbersome: hydra creates own build dir,
    # where data/ is not present. We need to escape to original cwd
    import hydra  # TODO(Przemek): do it better. Maybe download data?
    quickfix_path = Path(hydra.utils.get_original_cwd()) / filepath

    with open(quickfix_path) as f:
        return {line.strip() for line in f.readlines()}


def extract_data(
    rawdata_path: Path,
    male_attr_path: Path,
    female_attr_path: Path,
    stereo_target_path: Path,
    model_name: str,
    data_root: Path,
    num_proc: int,
) -> Dict[str, List[str]]:
    """Extracts, pre-processes and caches data.

    Extracts sentences that contain particular words: male&female attributes
    and stereotypes.

    Args:
        rawdata_path: path to a textfile with one sentence per line
        {male, female}_attr_path: textfile with one keyword per line
        stereo_target_path: textfile with one keyword per line
        model_name: used to instantiate a tokenizer.
        data_root: where to cache the data.
        num_proc: on how many processes run data pre-processing
    """
    # Get lists of attributes and targets
    male_attr = get_keyword_set(male_attr_path)
    female_attr = get_keyword_set(female_attr_path)
    stereo_trgt = get_keyword_set(stereo_target_path)

    tokenizer = Tokenizer(model_name)

    # This regexp basically tokenizes a sentence over spaces and 's, 're, 've..
    # It's originally taken from OpenAI's GPT-2 Encoder implementation
    pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # noqa E501

    np = 64
    # https://huggingface.co/docs/datasets/process.html#save

    dataset = load_dataset('text', data_files=rawdata_path, split="train[:]")
    dataset = dataset.map(
        lambda examples: get_keyword(examples, male_attr, female_attr, stereo_trgt, pattern),
        num_proc=np
    )
    dataset = dataset.filter(lambda examples: examples['type'] != 'none', num_proc=np)
    dataset = dataset.map(lambda examples: tokenizer(examples['text']), num_proc=np)
    dataset = dataset.map(lambda examples: get_keyword_mask(examples, tokenizer), num_proc=np)
    dataset = dataset.filter(lambda examples: sum(examples['keyword_mask']) > 0, num_proc=np)

    male = dataset.filter(lambda example: example['type'] == 'male', num_proc=np)
    female = dataset.filter(lambda example: example['type'] == 'female', num_proc=np)
    stereo = dataset.filter(lambda example: example['type'] == 'stereotype', num_proc=np)

    male = male.train_test_split(test_size=1000)
    female = female.train_test_split(test_size=1000)
    target = stereo.train_test_split(test_size=1000)

    male.set_format(type='torch', columns=['input_ids', 'attention_mask', 'keyword_mask'])
    female.set_format(type='torch', columns=['input_ids', 'attention_mask', 'keyword_mask'])
    target.set_format(type='torch', columns=['input_ids', 'attention_mask', 'keyword_mask'])

    male.save_to_disk(data_root / "male")
    female.save_to_disk(data_root / "female")
    target.save_to_disk(data_root / "stereotype")

    return {"male": male, "female": female, "stereotype": target}


def get_keyword(example, male_attr, female_attr, stereo_trgt, pattern):
    line = example['text']

    keywords = []

    line = line.strip()
    line_tokenized = {token.strip().lower() for token in re.findall(pattern, line)}

    # Dicts containing M/F/S attributes/targets in each sentence
    male = line_tokenized & male_attr
    female = line_tokenized & female_attr
    stereo = line_tokenized & stereo_trgt

    keyword_type = "none"

    # Note that a sentence might contain attributes of only one category
    #   M/F/S. That's why we check for emptiness of other sets.

    # Sentences with male attributes
    if len(male) > 0 and len(female) == 0:
        keywords.extend(male)
        keyword_type = "male"

    # Sentences with female attributes
    if len(female) > 0 and len(male) == 0:
        keywords.extend(female)
        keyword_type = "female"

    # Sentences with stereotype target
    if len(stereo) > 0 and len(male) == 0 and len(female) == 0:
        keywords.extend(stereo)
        keyword_type = "stereotype"

    example['keywords'] = keywords
    example['type'] = keyword_type

    return example


def get_keyword_mask(example, tokenizer):

    sentence_tokens = example['input_ids']
    keywords = ' '.join(example['keywords'])

    # Remove CLS/SEP and reshape, so it broadcasts nicely
    keyword_tokens = tokenizer(keywords, padding=False)
    keyword_tokens = torch.tensor(keyword_tokens['input_ids'])
    keyword_tokens = keyword_tokens[1:-1].reshape((-1, 1))

    sentence_tokens = torch.tensor(sentence_tokens).squeeze(0)

    # Mask indicating positions of attributes within sentence econdings
    # Each row of (sent==attr) contains position of consecutive tokens
    mask = (sentence_tokens == keyword_tokens).sum(0)

    example['keyword_mask'] = mask

    return example
