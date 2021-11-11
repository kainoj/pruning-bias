from typing import Any, Dict

from transformers.training_args import TrainingArguments
from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)

from src.models.debiaser import Debiaser


class DebiaserPruned(Debiaser):

    sparse_train_args: Dict[str, Any]

    def __post_init__(self):

        super().__post_init__()
        if self.model_name != 'bert-base-uncased':
            raise ValueError("Only bert-base-uncased is available for prunning.")

        sparse_train_args = SparseTrainingArguments(**self.sparse_train_args)

        self.model_patcher = ModelPatchingCoordinator(
            sparse_args=sparse_train_args,
            device=self.device,
            cache_dir='tmp/',
            model_name_or_path=self.model_name,
            logit_names='logits',
            teacher_constructor=None,
        )

        self.model_patcher.patch_model(self.model_debias.model)

        training_args = TrainingArguments(
            output_dir=None,  # Unused here, but required
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.model_patcher.create_optimizer_groups(
            self.model_debias.model,
            args=training_args,  # TODO: check values
            sparse_args=sparse_train_args
        )

    def forward(self, inputs, return_word_embs=None, embedding_layer=None):
        self.model_patcher.schedule_threshold(
            step=self.global_step,
            total_step=self.total_train_steps,
            training=False,  # TODO: pass a flag
        )

        return super().forward(inputs, return_word_embs, embedding_layer)

    def training_step(self, batch: Any, batch_idx: int):
        loss = super().training_step(batch, batch_idx)

        loss_prune_reg, _, _ = self.model_patcher.regularization_loss(self.model_debias.model)
        self.log(
            "train/loss/prune/regularize", loss_prune_reg,
            prog_bar=False, on_epoch=True, sync_dist=True
        )

        return loss + loss_prune_reg
