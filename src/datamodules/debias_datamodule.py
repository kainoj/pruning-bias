from dataclasses import dataclass
from typing import Dict
from pathlib import Path
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

from torch.utils.data import DataLoader
from torchtext.utils import download_from_url, extract_archive
from sklearn.model_selection import train_test_split

from src.models.modules.tokenizer import Tokenizer
from src.dataset.attributes_dataset import AttributesWithSentencesDataset
from src.dataset.targets_dataset import SentencesWithTargetsDatset
from src.dataset.utils import extract_data
from src.dataset.weat_dataset import WeatDataset
from src.metrics.seat import SEAT
from src.utils.utils import get_logger

import pickle

log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class DebiasDataModule(LightningDataModule):

    model_name: str
    batch_size: int
    data_dir: str
    datafiles: Dict[str, str]
    seat_data: Dict[str, str]

    # Filename to where cache data at
    cached_data_path: str

    def __post_init__(self):
        super().__init__()
        self.tokenizer = Tokenizer(self.model_name)
        self.data_dir = Path(self.data_dir)

        self.seat_dataset_map = {i: name for i, name in enumerate(self.seat_data.keys())}
        self.seat_metric = {name: SEAT() for name in self.seat_data.keys()}

    def prepare_data(self):
        extracted_path: Path  # One of the urls must be .gz

        for name, url in self.datafiles.items():
            print(url)
            datafiles = download_from_url(url, root=self.data_dir)
            self.datafiles[name] = datafiles
            if datafiles.endswith('.gz'):
                extracted_path = Path(datafiles).with_suffix('.txt').name
                extracted_path = extract_archive(datafiles, extracted_path)[0]

        # If data not cached, extract it and cache to a file
        if not Path(self.cached_data_path).exists():
            log.info(f'Extracting data from {extracted_path} '
                     f'and caching into {self.cached_data_path}')
            data = extract_data(
                rawdata_path=extracted_path,
                male_attr_path=self.datafiles['attributes_male'],
                female_attr_path=self.datafiles['attributes_female'],
                stereo_attr_path=self.datafiles['targets_stereotypes'],
                model_name=self.model_name
            )
            with open(self.cached_data_path, 'wb') as f:
                pickle.dump(data, f)

    def setup(self, stage):
        # Restore data from cache now
        log.info(f'Loading cached data from {self.cached_data_path}')
        with open(str(self.cached_data_path), 'rb') as f:
            data = pickle.load(f)

        # "We randomly sampled 1,000 sentences from each type of extracted
        # sentences as development data".
        # Here, Male&Female are our "attributes"
        m_train_sents, m_val_sents, m_train_attr, m_val_attr = train_test_split(
            data['male_sents'], data['male_sents_attr'], test_size=1000
        )
        f_train_sents, f_val_sents, f_train_attr, f_val_attr = train_test_split(
            data['female_sents'], data['female_sents_attr'], test_size=1000
        )
        # Steretypes are our "targets"
        s_train_sents, s_val_sents, s_train_trgt, s_val_trgt = train_test_split(
            data['stereo_sents'], data['stereo_sents_trgt'], test_size=1000
        )

        # Targets (stereotypes)
        self.targets_train = SentencesWithTargetsDatset(
            sentences=s_train_sents,
            targets_in_sentences=s_train_trgt,
            tokenizer=self.tokenizer
        )
        self.targets_val = self.data_val = SentencesWithTargetsDatset(
            sentences=s_val_sents,
            targets_in_sentences=s_val_trgt,
            tokenizer=self.tokenizer
        )

        # Attributes
        self.attributes_male_train = AttributesWithSentencesDataset(
            sentences=m_train_sents,
            attributes=m_train_attr,
            tokenizer=self.tokenizer
        )
        self.attributes_female_train = AttributesWithSentencesDataset(
            sentences=f_train_sents,
            attributes=f_train_attr,
            tokenizer=self.tokenizer
        )
        self.attributes_male_val = AttributesWithSentencesDataset(
            sentences=m_val_sents,
            attributes=m_val_attr,
            tokenizer=self.tokenizer
        )
        self.attributes_female_val = AttributesWithSentencesDataset(
            sentences=f_val_sents,
            attributes=f_val_attr,
            tokenizer=self.tokenizer
        )

        # SEAT Metric data (SEAT 6, 7 and 8)
        self.seat_datasets = {
            name: WeatDataset(data_filename=path, tokenizer=self.tokenizer)
            for name, path in self.seat_data.items()
        }

    def train_dataloader(self):
        """Train dataloader returns pairs of (target, attribute) embeddings."""
        targets = self.targets_train_dataloader()
        attributes = self.attributes_train_dataloader()
        return {"targets": targets, "attributes": attributes}

    def val_dataloader(self):
        """Validation dataloader returns pairs of (target, attribute) embeddings.

        `min_cycle` along with `shuffle=True` results in sampling a random
        subsets of equal size of male and female attributes, in every epoch.
        """
        targets = self.targets_val_dataloader()
        attributes = self.attributes_val_dataloader()
        return CombinedLoader(
            {"targets": targets, "attributes": attributes},
            "min_size"
        )

    # Targets
    def targets_train_dataloader(self):
        return DataLoader(
            dataset=self.targets_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def targets_val_dataloader(self):
        return DataLoader(
            dataset=self.targets_val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    # Attributes
    def attributes_train_dataloader(self):
        """This dataloader is used in the training step
        as well as in the computation fo non-contextualized embeddings.

        `min_cycle` along with `shuffle=True` results in sampling a random
        subsets of equal size of male and female attributes, in every epoch.
        """
        male = DataLoader(
            dataset=self.attributes_male_train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        female = DataLoader(
            dataset=self.attributes_female_train,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return CombinedLoader(
            {"male": male, "female": female},
            "min_size"
        )

    def attributes_val_dataloader(self):
        male = DataLoader(
            dataset=self.attributes_male_val,
            batch_size=self.batch_size,
            shuffle=False,
        )
        female = DataLoader(
            dataset=self.attributes_female_val,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return CombinedLoader(
            {"male": male, "female": female},
            "min_size"
        )

    # SEAT
    def seat_dataloaders(self):
        """Dataloaders for SEAT metrices (currently unused)."""
        return [
            DataLoader(ds, batch_size=1, shuffle=False) for ds in self.seat_datasets.values()
        ]
