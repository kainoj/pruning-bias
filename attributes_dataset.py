import urllib.request
import gzip
import shutil

from pathlib import Path

from torch.utils.data import Dataset


class AttributesDataset(Dataset):

    news_data_fname = 'news-commentary-v15.en.gz'
    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    def __init__(self, cache_dir='~/cache') -> None:
        super().__init__()

        self.cache_dir = self._prepare_cache_dir(cache_dir)
        
        self.prepare_data()

    def _prepare_cache_dir(self, cache_dir) -> Path:
        _cache_dir = Path(cache_dir).expanduser() / 'bs-data'
        Path(_cache_dir).mkdir(parents=True, exist_ok=True)
        return _cache_dir

    def prepare_data(self):

        data_fpath = self.cache_dir / self.news_data_fname

        if not data_fpath.exists():
            print(f'Dowloading data into {data_fpath}.')
            urllib.request.urlretrieve(self.news_data_url, data_fpath)
        else:
            print('Data exist, no need to download.')

        print(data_fpath)
        # with gzip.open(data_fpath, 'rt') as f_in:
        #     with gzip.open('/home/joe/file.txt.gz', 'wt') as f_out:
        #         shutil.copyfileobj(f_in, f_out)
        


if __name__ == "__main__":
    
    ds = AttributesDataset(cache_dir='~/bs/tmp')