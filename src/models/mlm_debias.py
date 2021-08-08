from typing import Any, List

import pickle
import torch
from pytorch_lightning import LightningModule

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.dataset.attributes_dataset import AttributesWithSentecesDataset
from src.dataset.targets_dataset import SentencesWithTargetsDatset
from src.dataset.utils import extract_data
from src.utils.utils import get_logger
from src.utils.data_io import download_and_un_gzip
from src.models.modules.mlm_pipeline import Pipeline


log = get_logger(__name__)

class MLMDebias(LightningModule):

    news_data_url = 'http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz'

    # Files with attribute lists
    female_attributes_filepath = 'data/female.txt'
    male_attributes_filepath = 'data/male.txt'
    stereotypes_filepath = 'data/stereotype.txt'

    # Filename to cache data at
    cached_data = 'attributes_dataset.obj'

    def __init__(
        self,
        model_name: str,
        get_embeddings_from: str,
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


        self.model = Pipeline(
            model_name=model_name,
            embeddings_from=get_embeddings_from
        )
        # For each attribute, keep its encdoing dict:
        # inputs_ids, attention_mask, attribute_indices
        self.sentences_of_attributes: List[dict[str, torch.tensor]]

    def on_train_epoch_start(self) -> None:
        # TODO the whole fun here
        # Surprisingly, in the paper they compute non-contextualized embeddings
        #  of atttributes *at the beginning of each epoch* 🤔
        log.info('Computing non-contextualized embeddings of attributes', self.device)

        for sents in self.attributes_dataloader():
            # print(sents)
            sents = {key: val.to(self.device) for key, val in sents.items()}

            for key, val in sents.items():
                print(f'key {key}  val shape.  {val.shape}')
            y = self(sents)       

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch: Any, batch_idx: int):
        # TODO: take care of types. Sentence must me a List[str], is a tuple
        # Possibly the solution would be to fix the dataset class return type
        sentences_with_targets = batch
        # sentences_of_attributes = self.sentences_of_attributes()
        y = self(sentences_with_targets)
        print(y.shape)


        loss = None  # TODO
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return None
    
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

        # TODO: move somewhere else
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # "We randomly sampled 1,000 sentences from each type of extracted sentences as development data.""
        # Male&Female are our "attributes"
        m_train_sents, m_val_sents, m_train_attr, m_val_attr = train_test_split(data['male_sents'], data['male_sents_attr'], test_size=1000)
        f_train_sents, f_val_sents, f_train_attr, f_val_attr = train_test_split(data['female_sents'], data['female_sents_attr'], test_size=1000)
        # Steretypes are our "targets"
        s_train_sents, s_val_sents, s_train_trgt, s_val_trgt = train_test_split(data['stereo_sents'], data['stereo_sents_trgt'], test_size=1000)

        train_sentences = s_train_sents
        train_targets_in_sentences = s_train_trgt

        self.data_train = SentencesWithTargetsDatset(
            sentences=train_sentences,
            targets_in_sentences=train_targets_in_sentences,
            tokenizer=tokenizer
        )
        self.data_val = SentencesWithTargetsDatset(
            sentences=s_val_sents,
            targets_in_sentences=s_val_trgt,
            tokenizer=tokenizer
        )

        attributes: List[str] = []
        sentences_of_attributes: List[List[str]] = []

        for attr, sents in data['attributes'].items():
            attributes.append(attr)
            sentences_of_attributes.append(sents)

        self.attributes_data = AttributesWithSentecesDataset(
            attributes=attributes,
            sentences_of_attributes=sentences_of_attributes,
            tokenizer=tokenizer
        )

        # Merge splitted M/F/S data into one
        # train_text = [*m_train_sents, *f_train_sents, *s_train_sents]
        # val_text = [*m_val_sents, *f_val_sents, *s_val_sents]

        # train_attr = [*m_train_attr, *f_train_attr, *s_train_attr]
        # val_attr = [*m_val_attr, *f_val_attr, *s_val_attr]

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

    def attributes_dataloader(self):
        return DataLoader(
            dataset=self.attributes_data,
            batch_size=self.batch_size,
            shuffle=False
        )