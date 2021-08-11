from typing import Any, List

import pickle
import torch
from pytorch_lightning import LightningModule

from pathlib import Path
from torch.autograd.grad_mode import no_grad
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset.attributes_dataset import AttributesWithSentecesDataset
from src.dataset.targets_dataset import SentencesWithTargetsDatset
from src.dataset.utils import extract_data
from src.utils.utils import get_logger
from src.utils.data_io import download_and_un_gzip
from src.models.modules.mlm_pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer
from src.metrics.seat import SEAT6, SEAT7, SEAT8

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

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
        embedding_layer: str,
        batch_size: int,
        data_dir: str,
        learning_rate: float,
        weight_decay: float,
        adam_eps: float,
        warmup_steps: int
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.warmup_steps = warmup_steps

        # Path to raw dataset, in format: /data/dir/news-commentary-v15.en.txt
        self.rawdata_path = (self.data_dir / Path(self.news_data_url).name).with_suffix('.txt')

        # Path to cached data (lists of attributes)
        self.cached_data_path = self.data_dir / self.cached_data

        self.model = Pipeline(
            model_name=model_name,
            embedding_layer=embedding_layer
        )

        self.tokenizer = Tokenizer(model_name)

        # Computed on the begining of each epoch
        self.non_contextualized: torch.tensor
    
        print("constructor")
        self.get_seat_scores()

    def get_seat_scores(self) -> None:
        
        def embbedder_fn(sentence):
            with torch.no_grad():
                tknzd = self.tokenizer(sentence).to(self.device)
                return self.model(tknzd, embedding_layer='CLS')

        seat_metrics = {"SEAT6": SEAT6(), "SEAT7": SEAT7(), "SEAT8": SEAT8()}

        for name in seat_metrics:
            value = seat_metrics[name](embbedder_fn)
            print(f'{name}: {value}')
        

    # def on_train_start(self) -> None:
    #     self.get_seat_scores()

    def on_train_epoch_start(self) -> None:

        self.get_seat_scores()
        
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

    def forward(self, inputs, return_word_embs=False):
        return self.model(inputs, return_word_embs)

    def loss(self, attributes, targets):
        attr = attributes.T               # (768, #attrs)
        trgt = targets.reshape((-1, 768))  # (bsz*128, 768)

        dot = torch.mm(trgt, attr)
        pow = torch.pow(dot, 2)

        return pow.sum()

    def training_step(self, batch: Any, batch_idx: int):

        targets = self(batch)
        attributes = self.non_contextualized

        loss = self.loss(attributes=attributes, targets=targets)

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):

        train_batches = len(self.train_dataloader()) // self.trainer.gpus
        total_epochs = self.trainer.max_epochs - self.trainer.min_epochs + 1
        total_train_steps = (total_epochs * train_batches) // self.trainer.accumulate_grad_batches

        no_decay = ["bias", "LayerNorm.weight"]

        # These parameters are copied from the original code
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": self.weight_decay,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }]

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

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
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
                stereo_attr_path=self.stereotypes_filepath,
                model_name=self.model_name
            )
            with open(str(self.cached_data_path), 'wb') as f:
                pickle.dump(data, f)

    def setup(self, stage):
        # Restore data from cache now
        log.info(f'Loading cached data from {self.cached_data_path}')
        with open(str(self.cached_data_path), 'rb') as f:
            data = pickle.load(f)

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