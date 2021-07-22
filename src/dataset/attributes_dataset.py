from typing import List, Tuple

import regex as re
import pickle

from pathlib import Path
from torch.utils.data import Dataset
from src.utils.utils import get_logger


log = get_logger(__name__)


class AttributesDataset(Dataset):

    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    cached_data = 'attributes_dataset.obj'

    def __init__(self, data_dir: str, rawdata: str) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.extract_data(rawdata)
    
    def get_attribute_set(self, filepath: str) -> set:
        """Reads file with attributes and returns a set containing them all"""

        # This is cumbersome: hydra creates own build dir,
        # where data/ is not present. We need to escape to original cwd
        import hydra  # TODO(Przemek): do it better. Maybe download data?
        quickfix_path = Path(hydra.utils.get_original_cwd()) / filepath

        with open(quickfix_path) as f:
            return {l.strip() for l in f.readlines()}

    def extract_data(self, rawdata: str) -> List[Tuple[str, str]]:
        cached_data_path = Path(self.data_dir) / self.cached_data

        if cached_data_path.exists():
            log.info(f'Loading cached data from {cached_data_path}')
            with open(str(cached_data_path), 'rb') as f:
                data = pickle.load(f)
        else:
            log.info(f'Extracting data from {rawdata} and caching into {cached_data_path}')
            data = self._extract_data(rawdata=rawdata)
            with open(str(cached_data_path), 'wb') as f:
                pickle.dump(data, f)

        return data
    
    def _extract_data(self, rawdata: Path) -> List[Tuple[str, str]]:

        female_attr = self.get_attribute_set(self.female_attributes_filepath)
        male_attr = self.get_attribute_set(self.male_attributes_filepath)
        stereo_attr = self.get_attribute_set(self.stereotypes_filepath)
        
        # This regexp basically tokenizes a sentence over spaces and 's, 're, 've..
        # It's originally taken from OpenAI's GPT-2 Encoder implementation
        pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        sentences = []

        with open(rawdata) as f:
            for iter, full_line in enumerate(f.readlines()):

                line = full_line.strip()

                if len(line) < 1 or len(line.split()) > 128 or len(line.split()) <= 1:
                    continue

                line_tokenized = {token.strip().lower() for token in re.findall(pat, line)}
                
                male = line_tokenized & male_attr
                female = line_tokenized & female_attr
                stereo = line_tokenized & stereo_attr

                if len(male) > 0 and len(female) == 0:
                    sentences.append(('M', male))
                
                if len(female) > 0 and len(male) == 0:
                    sentences.append(('F', female))
                    
                if len(stereo) > 0 and len(male) == 0 and len(female) == 0:
                    sentences.append(('S', stereo))

        return sentences
