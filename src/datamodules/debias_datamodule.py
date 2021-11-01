from dataclasses import dataclass
from typing import Dict
from pathlib import Path
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from datasets import load_from_disk
import torch

from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchtext.utils import download_from_url, extract_archive

from src.models.modules.tokenizer import Tokenizer
from src.dataset.utils import extract_data
from src.dataset.weat_dataset import WeatDataset
from src.metrics.seat import SEAT
from src.utils.utils import get_logger


log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class DebiasDataModule(LightningDataModule):

    model_name: str
    batch_size: int
    data_dir: str
    datafiles: Dict[str, str]
    seat_data: Dict[str, str]
    seed: int
    num_proc: int      # For dataset preprocessing
    num_workers: int   # For dataloaders

    def __post_init__(self):
        super().__init__()
        self.tokenizer = Tokenizer(self.model_name)
        self.data_dir = Path(self.data_dir)

        self.seat_metric = {name: SEAT() for name in self.seat_data.keys()}

        self.dataset_cache = self.data_dir / "dataset" / self.model_name / str(self.seed)

    def prepare_data(self):
        datafiles = {}
        for name, url in self.datafiles.items():
            download_path = download_from_url(url, root=self.data_dir)
            datafiles[name] = download_path
            if download_path.endswith('.gz'):
                extracted_path = extract_archive(download_path, self.data_dir)[0]
                datafiles[name] = extracted_path

        # The first call will cache the data
        if not self.dataset_cache.exists():
            log.info(f"Processing and caching the dataset to {self.dataset_cache}.")
            extract_data(
                rawdata_path=datafiles['plaintext'],
                male_attr_path=datafiles['attributes_male'],
                female_attr_path=datafiles['attributes_female'],
                stereo_target_path=datafiles['targets_stereotypes'],
                model_name=self.model_name,
                data_root=self.dataset_cache,
                num_proc=self.num_proc
            )
        else:
            log.info(f"Reading cached datset at {self.dataset_cache}")

    def setup(self, stage):
        # Data is cached to disk now
        data = {
            "male": load_from_disk(self.dataset_cache / "male"),
            "female": load_from_disk(self.dataset_cache / "female"),
            "target": load_from_disk(self.dataset_cache / "stereotype"),
        }

        for key in data:
            data[key].set_format(
                type='torch',
                columns=['input_ids', 'attention_mask', 'keyword_mask']
            )

        # Targets (stereotypes)
        self.targets_train = data['target']['train']
        self.targets_val = data['target']['test']

        # Attributes
        self.attributes_male_train = data['male']['train']
        self.attributes_female_train = data['female']['train']

        self.attributes_male_val = data['male']['test']
        self.attributes_female_val = data['female']['test']

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

        # If we pass a tensor with indices, 🤗datasets will fail -> .tolist()
        attributes_dataset = ConcatDataset([
            Subset(self.attributes_male_train, male_indices.tolist()),
            Subset(self.attributes_female_train, female_indices.tolist())
        ])

        attributes = DataLoader(
            dataset=attributes_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        targets = DataLoader(
            dataset=self.targets_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return {"targets": targets, "attributes": attributes}

    def val_dataloader(self):
        """Validation dataloader returns pairs of (target, attribute) embeddings.

        We don't balance attributes here, as we did in the trainlaoder
        (NB: they are already balanced).

        - M/F attributes contain 1k elements each -> 2k attributes.
        - There are 1k stereotype targets
        - With "max_size_cycle", this dataloader will make the 1st cycle on
          pairs (M, S), and the 2nd cycle on (F, S).
        """
        attributes_data = ConcatDataset([
            self.attributes_male_val,
            self.attributes_female_val,
        ])

        targets = DataLoader(
            dataset=self.targets_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        attributes = DataLoader(
            dataset=attributes_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return CombinedLoader(
            {"targets": targets, "attributes": attributes},
            "max_size_cycle"
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
