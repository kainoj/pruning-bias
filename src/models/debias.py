from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule

from tqdm import tqdm

from src.utils.utils import get_logger
from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer


from transformers import AdamW, get_linear_schedule_with_warmup

log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class Debiaser(LightningModule):

    model_name: str
    embedding_layer: str
    debias_mode: str  # for no only "sentence" TODO: "token"
    learning_rate: float
    weight_decay: float
    adam_eps: float
    warmup_steps: int
    loss_alpha: float
    loss_beta: float

    def __post_init__(self):
        super().__init__()

        self.model_debias = Pipeline(
            model_name=self.model_name,
            embedding_layer=self.embedding_layer
        )
        self.model_original = Pipeline(
            model_name=self.model_name,
            embedding_layer='all'
        )

        self.tokenizer = Tokenizer(self.model_name)

        # Computed on the begining of each epoch
        self.non_contextualized: torch.tensor = None

    def on_train_epoch_start(self) -> None:

        datamodule = self.trainer.datamodule

        log.info(f'Computing non-contextualized embeddings on'
                 f' {len(datamodule.attributes_data.sentences)} sentences.')

        non_contextualized_acc = torch.zeros((2, 768), device=self.device)
        non_contextualized_cntr = torch.zeros((2, 1), device=self.device)

        with torch.no_grad():
            for sents in tqdm(datamodule.attributes_dataloader()):

                sents = {key: val.to(self.device) for key, val in sents.items()}

                # Outputs contains only contextualized word embs for attributes
                outputs = self(sents, return_word_embs=True)

                attribute_ids = sents['attribute_gender']

                assert outputs.shape[0] == attribute_ids.shape[0]

                for attr_id, out in zip(attribute_ids, outputs):
                    non_contextualized_acc[attr_id] += out
                    non_contextualized_cntr[attr_id] += 1

            self.non_contextualized = non_contextualized_acc / non_contextualized_cntr
            self.non_contextualized.requires_grad_(False)

        log.info(f"Got non-contextualized embeddings of shape {self.non_contextualized.shape}")

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
        attr = static_attributes.T         # (768, #attrs)
        trgt = targets.reshape((-1, 768))  # (bsz*128, 768) # TODO get the dim

        # dot product -> sum rows -> square -> mean
        return torch.mm(trgt, attr).sum(1).pow(2).mean()

    def loss_regularize(self, attributes, attributes_original):
        """Loss for regularization (L2), Eq.(3)

        Args: contextualied embeddings of attributes, wrt to debiased
            and original model, respectively. Both are of shape:
                (batch_sz * n, emb_dim), where
            n = num_layers if embedding_layer=='all' else 1.
        """
        assert attributes.shape == attributes_original.shape
        return (attributes - attributes_original).pow(2).sum(1).mean()

    def step(self, batch) -> Dict[str, float]:
        """A step performed on training and validation.

        This is basically Eq.(4) in the paper.

        It computes debiasing loss with the regularizer term.
        """
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

        return {
            "loss": loss,
            "loss_debias": loss_debias,
            "loss_regularize": loss_regularize
        }

    def log_loss(self, loss: Dict[str, float], stage: str):
        """Loss logger for both training and validation.

        Args:
            loss: loss dict with keys: 'loss', 'loss_debias', 'loss_regularize'.
            stage: 'train'|'validation'
        """
        self.log(
            f"{stage}/loss/debias", loss["loss_debias"],
            prog_bar=False, on_epoch=True, sync_dist=True
        )
        self.log(
            f"{stage}/loss/regularize", loss["loss_regularize"],
            prog_bar=False, on_epoch=True, sync_dist=True
        )
        self.log(
            f"{stage}/loss", loss["loss"],
            prog_bar=True, on_epoch=True, sync_dist=True
        )

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log_loss(loss, 'train')
        return loss["loss"]

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def seat_step(self, batch: Any, batch_idx: int, dataset_idx: int):
        """SEAT step aka getting the metric update."""
        seat_name = self.seat_dataset_map[dataset_idx]

        target_x, target_y, attribute_a, attribute_b = batch
        self.seat_metric[seat_name].update(
            self(target_x, embedding_layer='CLS'),
            self(target_y, embedding_layer='CLS'),
            self(attribute_a, embedding_layer='CLS'),
            self(attribute_b, embedding_layer='CLS'),
        )

    def validation_step(self, batch: Any, batch_idx: int, dataset_idx: int):
        """In validations we have 4 datasets:
            * first three are for SEAT 6/7/8
            * the fourth is to get validation loss value
        """
        if dataset_idx < 3:
            self.seat_step(batch, batch_idx, dataset_idx)
        elif self.non_contextualized is not None:
            # On th sanity check the noncontextualized embeddings are not
            # initialized yet – effectively we'll compute only seat on sanity
            loss = self.step(batch)
            self.log_loss(loss, 'validation')

    def validation_epoch_end(self, outputs: List[Any]):
        for seat_name in self.seat_data.keys():
            seat_value = self.seat_metric[seat_name].compute()
            self.seat_metric[seat_name].reset()

            self.log(f"SEAT/{seat_name}", seat_value, sync_dist=True)

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
