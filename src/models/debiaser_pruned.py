import copy
from dataclasses import dataclass
from typing import Any, Dict

from transformers import AdamW, get_linear_schedule_with_warmup
from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)

from src.models.debiaser import Debiaser
from src.utils.utils import get_logger

log = get_logger(__name__)


class DebiaserPruned(Debiaser):

    sparse_train_args: Dict[str, Any]
    freeze_weights: bool
    share_pruning_scores: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.model_name != 'bert-base-uncased':
            raise ValueError("Only bert-base-uncased is available for prunning.")

        self.sparse_args = SparseTrainingArguments(**self.sparse_train_args)

        self.model_patcher = ModelPatchingCoordinator(
            sparse_args=self.sparse_args,
            device=self.device,
            cache_dir='tmp/',  # Used only for teacher
            model_name_or_path=self.model_name,
            logit_names='logits',  # TODO
            teacher_constructor=None,  # TODO
        )

        self.model_patcher.patch_model(self.model_debias.model)

        if self.freeze_weights:
            self.freeze_non_mask()

        if self.share_pruning_scores:
            self._share_pruning_scores()

    def freeze_non_mask(self):
        for name, param in self.model_debias.named_parameters():
            if name.split('.')[-1] != 'mask_scores':
                param.requires_grad = False

    def _share_pruning_scores(self):
        for layer in self.model_debias.model.encoder.layer:
            Qms = layer.attention.self.query.mask_module.context_modules[0].mask_scores.data
            layer.attention.self.key.mask_module.context_modules[0].mask_scores.data = Qms
            layer.attention.self.value.mask_module.context_modules[0].mask_scores.data = Qms

    def forward(self, inputs, return_word_embs=None, embedding_layer=None):
        self.model_patcher.schedule_threshold(
            step=self.global_step,
            total_step=self.total_train_steps,
            training=self.training,
        )
        return super().forward(inputs, return_word_embs, embedding_layer)

    def training_step(self, batch: Any, batch_idx: int):
        loss = super().training_step(batch, batch_idx)

        loss_prune_reg, _, _ = self.model_patcher.regularization_loss(self.model_debias.model)

        self.log(
            "train/loss/prune/regularize", loss_prune_reg,
            prog_bar=False, on_epoch=True, sync_dist=True
        )

        for stat, val in self.model_patcher.log().items():
            self.log(f"pruning/{stat}", val, prog_bar=False, on_epoch=False, sync_dist=True)

        return loss + loss_prune_reg

    def compile_model(self):
        """Returns compiled copy of a debiaed model (NOT in place)."""
        model = copy.deepcopy(self.model_debias.model)
        removed, heads = self.model_patcher.compile_model(model)

        log.info(f"Compiled model. Removed {removed} / {heads} heads.")

        return model

    def configure_optimizers(self):

        training_args = MockTrainingArgs(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        optim_groups = self.model_patcher.create_optimizer_groups(
            self.model_debias.model,
            args=training_args,  # TODO: check values
            sparse_args=self.sparse_args
        )

        optimizer = AdamW(optim_groups)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_train_steps
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


@dataclass
class MockTrainingArgs:
    learning_rate: float
    weight_decay: float
