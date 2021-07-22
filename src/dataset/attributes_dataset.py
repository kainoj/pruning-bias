from typing import Dict, List
import urllib.request
import gzip
import shutil
import regex as re
import pickle

from pathlib import Path

from torch.utils.data import Dataset


def download_and_un_gzip(url: str, path: Path) -> Path:
    """Downloads and uncompresses a .gz file.
    
    Args:
        url: url of a file to be downloaded. Must be a `.gz` file
        path: path to which uncompress the file
    
    Returns:
        path_txt: path to uncompressed file
    """

    filename = Path(url).name
    
    path_zip = path / filename
    path_txt = path_zip.with_suffix('.txt')

    if not path_zip.exists():
        print(f'Dowloading data into {path_zip}')
        urllib.request.urlretrieve(url, path_zip)
    
    if not path_txt.exists():
        print(f'Uncompressing data to {path_txt}')
        with gzip.open(path_zip, 'rt') as f_in:
            with open(path_txt, 'wt') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    return path_txt

def mkdir_if_not_exist(path: str, extend_path: str='') -> Path:
    """Makes a nested directory, if doesn't exist.
    
    Args:
        path: a root directory to make
        extend_path: whatever comes after `path`
    Returns: a path in form `path/extend_path`
    """
    _path = Path(path).expanduser() / extend_path
    Path(_path).mkdir(parents=True, exist_ok=True)
    return _path


class AttributesDataset(Dataset):

    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    data_pickle = 'attributes_dataset.obj'

    def __init__(self, cache_dir='~/cache') -> None:
        super().__init__()

        self.cache_dir = mkdir_if_not_exist(cache_dir, 'bs-data')
        self.data_txt_path = download_and_un_gzip(self.news_data_url, self.cache_dir)

        self.extract_data()
    
    def get_attribute_set(self, filepath: str) -> set:
        """Reads file with attributes and returns a set containing them all"""
        with open(filepath) as f:
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