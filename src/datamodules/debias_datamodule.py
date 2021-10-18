from dataclasses import dataclass
from typing import Dict
from pathlib import Path
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch

from torch.utils.data import DataLoader, ConcatDataset, Subset
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
        """Train dataloader returns pairs of (target, attribute) embeddings.

        There is an equal number of male and female attributes,
        sampled randomly from both sets.

        Train dataloader is reloaded on every epoch when
        `trainer.reload_dataloaders_every_n_epochs` is True.
        """
        assert self.trainer.reload_dataloaders_every_n_epochs == 1

        attr_len = min(len(self.attributes_male_train), len(self.attributes_female_train))

        male_indices = torch.randperm(len(self.attributes_male_train))[:attr_len]
        female_indices = torch.randperm(len(self.attributes_female_train))[:attr_len]

        attributes_dataset = ConcatDataset([
            Subset(self.attributes_male_train, male_indices),
            Subset(self.attributes_female_train, female_indices)
        ])

        attributes = DataLoader(
            dataset=attributes_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        targets = DataLoader(
            dataset=self.targets_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

        return {"targets": targets, "attributes": attributes}

    def val_dataloader(self):
        """Validation dataloader returns pairs of (target, attribute) embeddings.

        We don't balance attributes here, as we did in the trainlaoder
        (NB: they are already balanced).
        """
        attributes_data = ConcatDataset([
            self.attributes_male_val,
            self.attributes_female_val,
        ])

        targets = DataLoader(
            dataset=self.targets_val,
            batch_size=self.batch_size,
            shuffle=False,
        )
        attributes = DataLoader(
            dataset=attributes_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return CombinedLoader(
            {"targets": targets, "attributes": attributes},
            "min_size"  # TODO?
        )

    def attributes_train_dataloader(self):
        """This dataloader is used in the computation of non-contextualized embeddings.

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

    # SEAT
    def seat_dataloaders(self):
        """Dataloaders for SEAT metrices (currently unused)."""
        return [
            DataLoader(ds, batch_size=1, shuffle=False) for ds in self.seat_datasets.values()
        ]
