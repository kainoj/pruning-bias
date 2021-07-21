import urllib.request
import gzip
import shutil

from pathlib import Path

from torch.utils.data import Dataset


class AttributesDataset(Dataset):

    news_data_zip = 'news-commentary-v15.en.gz'
    news_data_txt = 'news-commentary-v15.en.txt'
    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    def __init__(self, cache_dir='~/cache') -> None:
        super().__init__()

        self.cache_dir = self._prepare_cache_dir(cache_dir)
        self.data_zip_path = self.cache_dir / self.news_data_zip
        self.data_txt_path = self.cache_dir / self.news_data_txt
        
        self.prepare_data()

    def _prepare_cache_dir(self, cache_dir) -> Path:
        _cache_dir = Path(cache_dir).expanduser() / 'bs-data'
        Path(_cache_dir).mkdir(parents=True, exist_ok=True)
        return _cache_dir

    def prepare_data(self):

        if not self.data_zip_path.exists():
            print(f'Dowloading data into {self.data_zip_path}')
            urllib.request.urlretrieve(self.news_data_url, self.data_zip_path)
        
        if not self.data_txt_path.exists():
            print(f'Extracting data to {self.data_txt_path}')
            with gzip.open(self.data_zip_path, 'rt') as f_in:
                with open(self.data_txt_path, 'wt') as f_out:
                    shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    
    ds = AttributesDataset(cache_dir='~/bs/tmp')