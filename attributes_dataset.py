from typing import Dict, List
import urllib.request
import gzip
import shutil
import regex as re

from pathlib import Path

from torch.utils.data import Dataset


class AttributesDataset(Dataset):

    news_data_zip = 'news-commentary-v15.en.gz'
    news_data_txt = 'news-commentary-v15.en.txt'
    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    def __init__(self, cache_dir='~/cache') -> None:
        super().__init__()

        self.cache_dir = self._prepare_cache_dir(cache_dir)
        self.data_txt_path = self.prepare_data()

        self.extract_data()

    def _prepare_cache_dir(self, cache_dir) -> Path:
        _cache_dir = Path(cache_dir).expanduser() / 'bs-data'
        Path(_cache_dir).mkdir(parents=True, exist_ok=True)
        return _cache_dir

    def prepare_data(self):

        data_zip_path = self.cache_dir / self.news_data_zip
        data_txt_path = self.cache_dir / self.news_data_txt

        if not data_zip_path.exists():
            print(f'Dowloading data into {data_zip_path}')
            urllib.request.urlretrieve(self.news_data_url, data_zip_path)
        
        if not data_txt_path.exists():
            print(f'Extracting data to {data_txt_path}')
            with gzip.open(data_zip_path, 'rt') as f_in:
                with open(data_txt_path, 'wt') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        return data_txt_path
    
    def get_attribute_set(self, filepath: str) -> set:
        """Reads file with attributes and returns a set containing them all"""
        with open(filepath) as f:
            return {l.strip() for l in f.readlines()}
    
    def extract_data(self) -> Dict[str, List[str]]:
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