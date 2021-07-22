from typing import Dict, List

import regex as re
import pickle

from pathlib import Path

from torch.utils.data import Dataset


class AttributesDataset(Dataset):

    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    data_pickle = 'attributes_dataset.obj'

    def __init__(self, chache_dir: Path) -> None:
        super().__init__()
        self.cache_dir = chache_dir
        self.extract_data()
    
    def get_attribute_set(self, filepath: str) -> set:
        """Reads file with attributes and returns a set containing them all"""

        # This is cumbersome: hydra creates own build dir,
        # where data/ is not present. We need to escape to original cwd
        import hydra  # TODO(Przemek): do it better. Maybe download data?
        quickfix_path = Path(hydra.utils.get_original_cwd()) / filepath

        with open(quickfix_path) as f:
            return {l.strip() for l in f.readlines()}

    def extract_data(self) -> Dict[str, List[str]]:
        pickle_path = self.cache_dir / self.data_pickle

        if pickle_path.exists():
            print("Loading data from cache")
            with open(str(pickle_path), 'rb') as f:
                data = pickle.load(f)
        else:
            print('Extracting and caching data')
            data = self._extract_data()
            with open(str(pickle_path), 'wb') as f:
                pickle.dump(data, f)

        return data
    
    def _extract_data(self) -> Dict[str, List[str]]:
        # if not cached -> extract and cache
        # if cached -> retrun cached

        female_attr = self.get_attribute_set(self.female_attributes_filepath)
        male_attr = self.get_attribute_set(self.male_attributes_filepath)
        stereo_attr = self.get_attribute_set(self.stereotypes_filepath)
        
        # This regexp basically tokenizes a sentence over spaces and 's, 're, 've..
        # It's originally taken from OpenAI's GPT-2 Encoder implementation
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        male_sentences = []
        female_sentences = []
        stereotype_sentences = []

        with open(self.data_txt_path) as f:
            for iter, full_line in enumerate(f.readlines()):

                line = full_line.strip()

                if len(line) < 1 or len(line.split()) > 128 or len(line.split()) <= 1:
                    continue

                line_tokenized = {token.strip().lower() for token in re.findall(pat, line)}
                
                male = line_tokenized & male_attr
                female = line_tokenized & female_attr
                stereo = line_tokenized & stereo_attr

                if len(male) > 0 and len(female) == 0:
                    male_sentences.append(male)
                
                if len(female) > 0 and len(male) == 0:
                    female_sentences.append(female)
                    
                if len(stereo) > 0 and len(male) == 0 and len(female) == 0:
                    stereotype_sentences.append(stereo)

        return {
            'male': male_sentences,
            'female': female_sentences,
            'stereotype': stereotype_sentences
        }

if __name__ == "__main__":
    
    ds = AttributesDataset(cache_dir='~/bs/tmp')