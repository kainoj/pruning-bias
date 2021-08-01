from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.dataset.attributes_dataset import AttributesDataset, extract_data

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

        #  We randomly sampled 1,000 sentences from each type of extracted sentences as development data.
        m_train_sents, m_val_sents, m_train_attr, m_val_attr = train_test_split(data['male_sents'], data['male_sents_attr'], test_size=1000)
        f_train_sents, f_val_sents, f_train_attr, f_val_attr = train_test_split(data['female_sents'], data['female_sents_attr'], test_size=1000)
        s_train_sents, s_val_sents, s_train_attr, s_val_attr = train_test_split(data['stereo_sents'], data['stereo_sents_attr'], test_size=1000)

        # Merge splitted M/F/S data into one
        train_text = [*m_train_sents, *f_train_sents, *s_train_sents]
        val_text = [*m_val_sents, *f_val_sents, *s_val_sents]

        train_attr = [*m_train_attr, *f_train_attr, *s_train_attr]
        val_attr = [*m_val_attr, *f_val_attr, *s_val_attr]

        attr2sents = data['attributes']

        # TODO: move somewhere else
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.data_train = AttributesDataset(train_text, train_attr, attr2sents, tokenizer)
        self.data_val = AttributesDataset(val_text, val_attr, attr2sents, tokenizer)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            # We don't really need workers for in-mem data-right?
            # num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            # collate_fn=lambda x: x,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            # We don't really need workers for in-mem data-right?
            # num_workers=self.num_workers,
            # pin_memory=self.pin_memory,
            # collate_fn=lambda x: x,
            shuffle=False,
        )
