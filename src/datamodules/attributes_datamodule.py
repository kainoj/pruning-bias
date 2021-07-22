from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from src.dataset.attributes_dataset import AttributesDataset

from src.utils.utils import get_logger
from src.utils.data_io import download_and_un_gzip, mkdir_if_not_exist

log = get_logger(__name__)


class AttributesDataModule(LightningDataModule):

    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    def __init__(
        self,
        batch_size: int,
        data_dir: str
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir

    def prepare_data(self):
        rawdata = download_and_un_gzip(self.news_data_url, self.data_dir)

        # The first call will download, extract and cache data
        AttributesDataset(data_dir=self.data_dir, rawdata=rawdata)

    def setup(self, stage):
        # Data will be recovered from cache now
        dataset = AttributesDataset(data_dir=self.data_dir, rawdata=None)
        data_size = len(dataset)

        #  We randomly sampled 1,000 sentences from each type of
        #   extracted sentences as development data. TODO: do it better!
        self.data_train, self.data_val = random_split(
            dataset, [data_size - 1000, 1000]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            # We don't really need workers for in-mem data-right?
            # num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            # We don't really need workers for in-mem data-right?
            # num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            shuffle=False,
        )
