import hydra
import torch
from omegaconf import DictConfig

from src.models.modules.mlm_pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer


@hydra.main(config_path="configs", config_name="eval")
def evaluate(cfg: DictConfig) -> None:
    from src.metrics.seat import SEAT6, SEAT7, SEAT8

    seat_metrics = {"SEAT6": SEAT6(), "SEAT7": SEAT7(), "SEAT8": SEAT8()}
    device = 'cuda:0'  # move to config

    for model_name in cfg.models:
        print(f"Evaluating: '{model_name}'")

        pipeline = Pipeline(model_name, embedding_layer='last').to(device)
        tokenizer = Tokenizer(model_name)

        # A quick workaround, we need something callable
        #   which takes str and returns an embedding.
        def embbedder_fn(sentence):
            with torch.no_grad():
                tknzd = tokenizer(sentence).to(device)
                return pipeline(tknzd, embedding_layer='CLS')

        embedder = embbedder_fn

        for name in seat_metrics:
            value = seat_metrics[name](embedder)
            print(f"{name}: {value}")


if __name__ == "__main__":
    evaluate()
