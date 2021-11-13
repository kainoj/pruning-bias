import copy

from pathlib import Path
from src.utils.utils import get_logger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

log = get_logger(__name__)


class ModelCheckpointWithHuggingface(ModelCheckpoint):

    def __init__(self, compile_pruned=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_pruned = compile_pruned

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):

        # Save Lightning checkpoint (everything)
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Save debiased model and tokenizer only (🤗 compatibile)
        # filename = epoch=X-step=X
        filename = self._format_checkpoint_name(
            None, dict(epoch=trainer.current_epoch, step=trainer.global_step)
        )

        path = Path(self.dirpath) / "debias" / filename

        pl_module.tokenizer._tokenizer.save_pretrained(path)

        if not self.compile_pruned:
            pl_module.model_debias.model.save_pretrained(path)
        else:
            # Because model compile is in place
            model = copy.deepcopy(pl_module.model_debias.model)
            removed, heads = pl_module.model_patcher.compile_model(model)

            # TODO: save spars_args.json, trainin_args.json, ...
            model.save_pretrained(path)

            log.info(f"Compiled model. Removed {removed} / {heads} heads.")
