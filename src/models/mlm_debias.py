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
        # For each attribute, keep its encdoing dict:
        # inputs_ids, attention_mask, attribute_indices
        self.sentences_of_attributes: List[dict[str, torch.tensor]]

    def on_train_start(self) -> None:
        return
        print('Getting attributes....')
        # Watch out! Since we're bypassing dataloader, i.e. accessing data
        #  directly from dataset, we have to manage data device manually.
        #  TODO: We can actually wrap it into a dataset and/or dataloader
        ds = self.train_dataloader().dataset
        self.sentences_of_attributes = ds.get_attributes_with_sentences()

    def on_train_epoch_start(self) -> None:
        return
        # TODO the whole fun here
        # Surprisingly, in the paper they compute non-contextualized embeddings
        #  of atttributes at the beginning of each epoch 🤔
        print('getting the attribute static', self.device)

        for sents in self.sentences_of_attributes:
            # shiet it's not the best idea to feed 1k+ sentences...
            sents.to(self.device)
            print(sents.keys())
            y = self(sents)

        return super().on_train_epoch_start()

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