import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.sentence_embbeder import SentenceEmbedder
from src.metrics.seat import SEAT6, SEAT7, SEAT8


@hydra.main(config_path="configs", config_name="eval")
def evaluate(cfg: DictConfig) -> None:
    
    seat_metrics = {"SEAT6": SEAT6(), "SEAT7": SEAT7(), "SEAT8": SEAT8()}
      
    for model_name in cfg.models:
        print(f"Evaluating: '{model_name}'")
        embedder = SentenceEmbedder(model_name)

        for name in seat_metrics:
            value = seat_metrics[name](embedder)
            print(f"{name}: {value}")


if __name__ == "__main__":
    evaluate()

    
