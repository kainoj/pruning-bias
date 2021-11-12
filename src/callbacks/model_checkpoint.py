from pathlib import Path
from src.utils.utils import get_logger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

log = get_logger(__name__)


class ModelCheckpointWithHuggingface(ModelCheckpoint):

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):

        # Save Lightning checkpoint (everything)
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Save debiased model and tokenizer only (🤗 compatibile)
        # filename = epoch=X-step=X
        filename = self._format_checkpoint_name(
            None, dict(epoch=trainer.current_epoch, step=trainer.global_step)
        )

        path = Path(self.dirpath) / "debias" / filename

        pl_module.model_debias.model.save_pretrained(path)
        pl_module.tokenizer._tokenizer.save_pretrained(path)
