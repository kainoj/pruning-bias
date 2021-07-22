from typing import Any, List

import torch
from pytorch_lightning import LightningModule

class MLMDebias(LightningModule):

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass

    def training_step(self, batch: Any, batch_idx: int):
        pass

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return None