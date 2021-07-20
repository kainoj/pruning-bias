import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.sentence_embbeder import SentenceEmbedder
from src.metrics.seat import SEAT6, SEAT7, SEAT8


@hydra.main(config_path="configs", config_name="eval")
def evaluate(cfg: DictConfig) -> None:
    
    model_name = cfg.models[0]
    
    embedder = SentenceEmbedder(model_name)

    seat_metrics = {"SEAT6": SEAT6(), "SEAT7": SEAT7(), "SEAT8": SEAT8()}

    for name in seat_metrics:
        value = seat_metrics[name](embedder)
        print(f"{name}: {value}")


if __name__ == "__main__":
    evaluate()

    
