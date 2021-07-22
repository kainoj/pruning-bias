from src.dataset.attributes_dataset import AttributesDataset

from pytorch_lightning import LightningDataModule


class AttributesDataModule(LightningDataModule):

    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        # train_split = Dataset(...)
        # return DataLoader(train_split)
        pass
    
    def val_dataloader(self):
        # val_split = Dataset(...)
        # return DataLoader(val_split)
        pass

    def test_dataloader(self):
        # test_split = Dataset(...)
        # return DataLoader(test_split)
        pass

    def teardown(self):
        pass