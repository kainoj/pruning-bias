from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, ConcatDataset

from src.dataset.attributes_dataset import AttributesDataset, extract_data, get_attribute_set

from src.utils.utils import get_logger
from src.utils.data_io import download_and_un_gzip

import pickle

log = get_logger(__name__)


class AttributesDataModule(LightningDataModule):

    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    # Files with attribute lists
    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    # Filename to cache data at
    cached_data = 'attributes_dataset.obj'

    def __init__(
        self,
        batch_size: int,
        data_dir: str
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = Path(data_dir)

        # Path to raw dataset, in format: /data/dir/news-commentary-v15.en.txt
        self.rawdata_path = (self.data_dir / Path(self.news_data_url).name).with_suffix('.txt')

        # Path to cached data (lists of attributes)
        self.cached_data_path = self.data_dir / self.cached_data

        print("RAW DATA DIR", self.rawdata_path, self.cached_data_path)

    def prepare_data(self):
        # Download and unzip the News dataset
        download_and_un_gzip(self.news_data_url, self.rawdata_path)

        # If data not cached, extract it and cache to a file
        if not self.cached_data_path.exists():
            log.info(f'Extracting data from {self.rawdata_path} and caching into {self.cached_data_path}')
            data = extract_data(
                rawdata_path=self.rawdata_path,
                male_attr_path=self.male_attributes_filepath,
                female_attr_path=self.female_attributes_filepath,
                stereo_attr_path=self.stereotypes_filepath
            )
            with open(str(self.cached_data_path), 'wb') as f:
                pickle.dump(data, f)

    def setup(self, stage):
        # Restore data from cache now
        log.info(f'Loading cached data from {self.cached_data_path}')
        with open(str(self.cached_data_path), 'rb') as f:
            data = pickle.load(f)
        
        # Make one dataset for each subset, so we can easily do train/dev splits
        ds_male = AttributesDataset(sentences=data['male'])
        ds_female = AttributesDataset(sentences=data['female'])
        ds_stereo = AttributesDataset(sentences=data['stereo'])

        attr2sents = data['attributes']

        #  We randomly sampled 1,000 sentences from each type of
        #   extracted sentences as development data.
        male_train, male_val = random_split(ds_male, [len(ds_male) - 1000, 1000])
        female_train, female_val = random_split(ds_female, [len(ds_female) - 1000, 1000])
        stereo_train, stereo_val = random_split(ds_stereo, [len(ds_stereo) - 1000, 1000])

        self.data_train = ConcatDataset([male_train, female_train, stereo_train])
        self.data_val = ConcatDataset([male_val, female_val, stereo_val])


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
