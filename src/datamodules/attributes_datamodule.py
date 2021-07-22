from pathlib import Path
from pytorch_lightning import LightningDataModule

from src.dataset.attributes_dataset import AttributesDataset

from src.utils.utils import get_logger
from src.utils.data_io import download_and_un_gzip, mkdir_if_not_exist

log = get_logger(__name__)


class AttributesDataModule(LightningDataModule):

    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    def __init__(self):
        super().__init__()
        print("hello module")

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        chache_dir = mkdir_if_not_exist('~/bs/tmp', 'bs-data')
        rawdata = download_and_un_gzip(self.news_data_url, chache_dir)

        ds = AttributesDataset(chache_dir=chache_dir, rawdata=rawdata)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        # train_split = Dataset(...)
        # return DataLoader(train_split)
        pass
    
    def val_dataloader(self):
        # val_split = Dataset(...)
        # return DataLoader(val_split)
        pass

    def test_dataloader(self):
        # test_split = Dataset(...)
        # return DataLoader(test_split)
        pass

    def teardown(self):
        pass