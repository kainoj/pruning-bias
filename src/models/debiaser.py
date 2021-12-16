from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule

from tqdm import tqdm
from src.metrics.seat import SEAT

from src.utils.utils import get_logger
from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer


from transformers import AdamW, get_linear_schedule_with_warmup

log = get_logger(__name__)


@dataclass(unsafe_hash=True)
class Debiaser(LightningModule):

    model_name: str
    embedding_layer: str
    debias_mode: str
    learning_rate: float
    weight_decay: float
    adam_eps: float
    warmup_steps: int
    loss_alpha: float
    loss_beta: float
    hf_checkpoint: str = None
    is_glue: bool = False

    # Used by child only
    sparse_train_args: Dict[str, Any] = None
    freeze_weights: bool = False
    share_pruning_scores: bool = False
    prune_values_only: bool = False
    prune_attention_only: bool = False

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model_debias = Pipeline(
            model_name=self.model_name,
            embedding_layer=self.embedding_layer,
            debias_mode=self.debias_mode,
            hf_checkpoint=self.hf_checkpoint,
            is_glue=self.is_glue
        )
        self.model_original = Pipeline(
            model_name=self.model_name,
            embedding_layer='all',   # See Eq. (3)
            debias_mode='sentence',  # See Eq. (3)
            hf_checkpoint=self.hf_checkpoint,
            is_glue=self.is_glue
        )

        self.tokenizer = Tokenizer(self.model_name)

        # Non-contextualized embeddings are computed on the begining of each epoch
        self.non_contextualized: torch.tensor = None

    def on_train_epoch_start(self) -> None:
        self.non_contextualized = self.get_non_contextualized()

    def on_validation_start(self):
        self.compute_seat()
        if self.non_contextualized is None:
            # This will happen only on the very first val run,
            # before the training even begins.
            self.non_contextualized = self.get_non_contextualized()
            log.info("Evaluating vanilla pre-trained model")

    def get_non_contextualized(self) -> torch.tensor:
        """Returns only two embeddings, each being an average of male (female)
        related attributes.
        """
        log.info('Computing non-contextualized embeddings...')

        datamodule = self.trainer.datamodule

        non_contextualized_acc = torch.zeros((2, self.model_debias.dim), device=self.device)
        non_contextualized_cntr = torch.zeros((2, 1), device=self.device)

        mode = self.training
        self.eval()
        with torch.no_grad():
            for batch in tqdm(datamodule.attributes_train_dataloader()):

                m_sents = {key: val.to(self.device) for key, val in batch['male'].items()}
                f_sents = {key: val.to(self.device) for key, val in batch['female'].items()}

                for i, sents in enumerate([m_sents, f_sents]):
                    # Outputs contains only contextualized word embs for attributes
                    outputs = self(sents, return_word_embs=True)

                    non_contextualized_acc[i] += outputs.sum(dim=0)
                    non_contextualized_cntr[i] += outputs.shape[0]

            non_contextualized = non_contextualized_acc / non_contextualized_cntr
            non_contextualized.requires_grad_(False)

        self.train(mode)  # Restore original mode

        log.info(f"Got non-contextualized embeddings of shape {non_contextualized.shape}")

        return non_contextualized

    def forward(self, inputs, return_word_embs=None, embedding_layer=None):
        """Forward pass of the model to be debiased."""
        return self.model_debias(inputs, return_word_embs, embedding_layer)

    def forward_original(self, inputs, return_word_embs=None, embedding_layer=None):
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
        attr = static_attributes.T                           # (dim, #attrs)
        trgt = targets.reshape((-1, self.model_debias.dim))  # (bsz*128, dim)

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

        Note, that in the regularization term, *word* embeddings are taken
        across *all* layers in both models (see Eq. 3).
        """
        targets = self(batch["targets"])

        attributes = self(
            batch['attributes'], return_word_embs=True, embedding_layer='all'
        )
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
            prog_bar=False, on_epoch=True, sync_dist=True
        )

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log_loss(loss, 'train')
        return loss["loss"]

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def compute_seat(self):
        """Evaluates the SEAT scores for each available metric."""
        def to_device(inputs: dict[str, torch.tensor]):
            return {key: val.to(self.device) for key, val in inputs.items()}

        for seat_name, dataset in self.trainer.datamodule.seat_datasets.items():
            seat = SEAT()
            data = dataset.get_all_items()

            target_x = to_device(data['target_x'])
            target_y = to_device(data['target_y'])
            attribute_a = to_device(data['attribute_a'])
            attribute_b = to_device(data['attribute_b'])

            mode = self.training
            self.eval()
            with torch.no_grad():
                value = seat(
                    self(target_x, embedding_layer='CLS'),
                    self(target_y, embedding_layer='CLS'),
                    self(attribute_a, embedding_layer='CLS'),
                    self(attribute_b, embedding_layer='CLS'),
                )
            self.train(mode)  # Restore training mode

            self.log(f"SEAT/{seat_name}", value, sync_dist=True)

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log_loss(loss, 'validation')

    @property
    def total_train_steps(self):
        num_devices = 1
        if self.trainer.gpus and self.trainer.gpus > 0:
            if isinstance(self.trainer.gpus, list):
                num_devices = len(self.trainer.gpus)
            else:
                num_devices = self.trainer.gpus

        # Be carefull: trainloader is a dict of loaders of equal length
        num_samples = len(self.train_dataloader()["targets"])
        train_batches = num_samples // num_devices
        total_epochs = self.trainer.max_epochs - self.trainer.min_epochs + 1

        return (total_epochs * train_batches) // self.trainer.accumulate_grad_batches

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model_debias.parameters(),
            weight_decay=self.weight_decay,
            lr=self.learning_rate,
            eps=self.adam_eps
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_train_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
