from typing import Any, List

import torch
from pytorch_lightning import LightningModule

from src.models.modules.mlm_pipeline import Pipeline


class MLMDebias(LightningModule):

    def __init__(
        self,
        model_name: str,
        get_embeddings_from: str
    ) -> None:
        super().__init__()

        self.model = Pipeline(
            model_name=model_name,
            embeddings_from=get_embeddings_from
        )

    def forward(self, x: List[str]):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        # TODO: take care of types. Sentence must me a List[str], is a tuple
        # Possibly the solution would be to fix the dataset class return type
        _, sent = batch 
        y = self(sent) # calls forward

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