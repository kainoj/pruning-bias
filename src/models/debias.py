from dataclasses import dataclass
from typing import Any, Dict, List

import pickle
import torch
from pytorch_lightning import LightningModule

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset.attributes_dataset import AttributesWithSentecesDataset
from src.dataset.targets_dataset import SentencesWithTargetsDatset
from src.dataset.utils import extract_data
from src.dataset.weat_dataset import WeatDataset
from src.utils.utils import get_logger
from src.utils.data_io import download_and_un_gzip
from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer
from src.metrics.seat import SEAT


from transformers import AdamW, get_linear_schedule_with_warmup

log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class Debiaser(LightningModule):

    model_name: str
    embedding_layer: str
    debias_mode: str  # for no only "sentence" TODO: "token"
    batch_size: int
    data_dir: str
    learning_rate: float
    weight_decay: float
    adam_eps: float
    warmup_steps: int
    loss_alpha: float
    loss_beta: float
    seat_data: Dict[str, str]

    news_data_url: str

    # Files with predefined attributes, one per line
    female_attributes_filepath: str
    male_attributes_filepath: str
    stereotypes_filepath: str

    # Filename to where cache data at
    cached_data_path: str

    def __post_init__(self):
        super().__init__()

        self.seat_dataset_map = {i: name for i, name in enumerate(self.seat_data.keys())}

        self.data_dir = Path(self.data_dir)

        # Path to raw dataset, in format: /data/dir/news-commentary-v15.en.txt
        self.rawdata_path = (self.data_dir / Path(self.news_data_url).name).with_suffix('.txt')

        self.model_debias = Pipeline(
            model_name=self.model_name,
            embedding_layer=self.embedding_layer
        )
        self.model_original = Pipeline(
            model_name=self.model_name,
            embedding_layer='all'
        )

        self.tokenizer = Tokenizer(self.model_name)

        # Create a metric for each of provided datasets
        self.seat_metric = {name: SEAT() for name in self.seat_data.keys()}

        # Computed on the begining of each epoch
        self.non_contextualized: torch.tensor

    def on_train_epoch_start(self) -> None:

        log.info(f'Computing non-contextualized embeddings of'
                 f' {len(self.attributes_data.attributes)} attributes on'
                 f' {len(self.attributes_data.sentences)} sentences.')

        num_attrs = len(self.attributes_data.attributes)
        non_contextualized_acc = torch.zeros((num_attrs, 768), device=self.device)
        non_contextualized_cntr = torch.zeros((num_attrs, 1), device=self.device)

        with torch.no_grad():
            for sents in tqdm(self.attributes_dataloader()):

                sents = {key: val.to(self.device) for key, val in sents.items()}

                # Outputs contains only contextualized word embs for attributes
                outputs = self(sents, return_word_embs=True)

                attribute_ids = sents['attribute_id']

                assert outputs.shape[0] == attribute_ids.shape[0]

                # Ups, this won't work if values of attribute_ids are not distinct 🤷🏼‍♂️
                # non_contextualized_acc[attribute_ids] += outputs
                # non_contextualized_cntr[attribute_ids] += 1

                # A quick workaround:
                for attr_id, out in zip(attribute_ids, outputs):

                    non_contextualized_acc[attr_id] += out
                    non_contextualized_cntr[attr_id] += 1

            self.non_contextualized = non_contextualized_acc / non_contextualized_cntr
            self.non_contextualized.requires_grad_(False)

    def forward(
        self, inputs, return_word_embs=False, embedding_layer=None
    ):
        """Forward pass of the models to be debiased."""
        return self.model_debias(inputs, return_word_embs, embedding_layer)

    def forward_original(
        self, inputs, return_word_embs=False, embedding_layer=None
    ):
        """Forward pass of the original model (frozen)."""
        with torch.no_grad():
            return self.model_original(inputs, return_word_embs, embedding_layer)

    def loss_debias(self, static_attributes, targets):
        """Loss for debiasing (inner product), Eq.(1)

        Args:
            attributes: NON-CONTEXTUALIZED (aka static) embeddings of attributes
                that were precomputed at the beginning of the epoch
            targets: contextualized embeddigs of targets of current batch
        """
        attr = static_attributes.T                # (768, #attrs)
        trgt = targets.reshape((-1, 768))  # (bsz*128, 768) # TODO get the dim

        dot = torch.mm(trgt, attr) ** 2

        # Sum across rows,then take mean
        return dot.sum(1).mean()

    def loss_regularize(self, attributes, attributes_original):
        """Loss for regularization (L2), Eq.(3)

        Args: contextualied embeddings of attributes, wrt to debiased
            and original model, respectively. Both are of shape:
                (batch_sz * n, emb_dim), where
            n = num_layers if embedding_layer=='all' else 1.
        """
        assert attributes.shape == attributes_original.shape
        return ((attributes - attributes_original) ** 2).sum(1).mean()

    def training_step(self, batch: Any, batch_idx: int):

        targets = self(batch["targets"])
        attributes = self(batch['attributes'], return_word_embs=True, embedding_layer='all')
        attributes_original = self.forward_original(
            batch['attributes'], return_word_embs=True, embedding_layer='all'
        )

        loss_debias = self.loss_debias(
            static_attributes=self.non_contextualized, targets=targets
        )
        loss_regularize = self.loss_regularize(
            attributes=attributes, attributes_original=attributes_original
        )

        loss = self.loss_alpha * loss_debias + self.loss_beta * loss_regularize

        self.log("train/loss/debias", loss_debias, prog_bar=False, on_epoch=True)
        self.log("train/loss/regularize", loss_regularize, prog_bar=False, on_epoch=True)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int, dataset_idx: int):
        # Get the SEAT
        seat_name = self.seat_dataset_map[dataset_idx]

        target_x, target_y, attribute_a, attribute_b = batch
        self.seat_metric[seat_name].update(
            self(target_x, embedding_layer='CLS'),
            self(target_y, embedding_layer='CLS'),
            self(attribute_a, embedding_layer='CLS'),
            self(attribute_b, embedding_layer='CLS'),
        )
        return 42

    def validation_epoch_end(self, outputs: List[Any]):
        for seat_name in self.seat_data.keys():
            seat_value = self.seat_metric[seat_name].compute()
            self.seat_metric[seat_name].reset()

            self.log(f"validation/{seat_name}", seat_value)

    def configure_optimizers(self):
        train_batches = len(self.train_dataloader()) // self.trainer.gpus
        total_epochs = self.trainer.max_epochs - self.trainer.min_epochs + 1
        total_train_steps = (total_epochs * train_batches) // self.trainer.accumulate_grad_batches

        no_decay = ["bias", "LayerNorm.weight"]

        # These parameters are copied from the original code
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model_debias.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model_debias.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=self.adam_eps
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_train_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def prepare_data(self):
        # Download and unzip the News dataset
        download_and_un_gzip(self.news_data_url, self.rawdata_path)

        # If data not cached, extract it and cache to a file
        if not Path(self.cached_data_path).exists():
            log.info(f'Extracting data from {self.rawdata_path} '
                     f'and caching into {self.cached_data_path}')
            data = extract_data(
                rawdata_path=self.rawdata_path,
                male_attr_path=self.male_attributes_filepath,
                female_attr_path=self.female_attributes_filepath,
                stereo_attr_path=self.stereotypes_filepath,
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
        # sentences as development data". Here Male&Female are our "attributes"
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

        train_sentences = s_train_sents
        train_targets_in_sentences = s_train_trgt

        self.data_train = SentencesWithTargetsDatset(
            sentences=train_sentences,
            targets_in_sentences=train_targets_in_sentences,
            tokenizer=self.tokenizer
        )
        self.data_val = SentencesWithTargetsDatset(
            sentences=s_val_sents,
            targets_in_sentences=s_val_trgt,
            tokenizer=self.tokenizer
        )

        attributes: List[str] = []
        sentences_of_attributes: List[List[str]] = []

        for attr, sents in data['attributes'].items():
            attributes.append(attr)
            sentences_of_attributes.append(sents)

        self.attributes_data = AttributesWithSentecesDataset(
            attributes=attributes,
            sentences_of_attributes=sentences_of_attributes,
            tokenizer=self.tokenizer
        )

        # TODO: rename it whatever
        self.seat_datasets = {
            name: WeatDataset(data_filename=path, tokenizer=self.tokenizer)
            for name, path in self.seat_data.items()
        }

        # Merge splitted M/F/S data into one
        # train_text = [*m_train_sents, *f_train_sents, *s_train_sents]
        # val_text = [*m_val_sents, *f_val_sents, *s_val_sents]

        # train_attr = [*m_train_attr, *f_train_attr, *s_train_attr]
        # val_attr = [*m_val_attr, *f_val_attr, *s_val_attr]

    def train_dataloader(self):
        # TODO: what about num workers?
        targets = DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        attributes = DataLoader(
            dataset=self.attributes_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        return {"targets": targets, "attributes": attributes}

    def val_dataloader(self):
        """For now, returns only data for SEAT6

        It passes batches sequentally.
        https://pytorch-lightning.readthedocs.io/en/latest/guides/data.html#multiple-validation-test-datasets
        """
        return [
            DataLoader(ds, batch_size=1, shuffle=False) for ds in self.seat_datasets.values()
        ]

    def attributes_dataloader(self):
        return DataLoader(
            dataset=self.attributes_data,
            batch_size=self.batch_size,
            shuffle=False
        )
