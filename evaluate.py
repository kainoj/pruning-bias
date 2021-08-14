import hydra
import torch

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer
from src.dataset.weat_dataset import WeatDataset


@hydra.main(config_path="configs", config_name="eval")
def evaluate(cfg: DictConfig) -> None:
    from src.metrics.seat import SEAT

    seat_data = {
        "SEAT6": '/home/przm/bs/data/sent-weat6.jsonl',
        "SEAT7": '/home/przm/bs/data/sent-weat7.jsonl',
        "SEAT8": '/home/przm/bs/data/sent-weat8.jsonl',
    }

    device = cfg.device

    for model_name in cfg.models:
        print(f"Evaluating: '{model_name}'")

        model = Pipeline(model_name, embedding_layer='CLS').to(device)
        tokenizer = Tokenizer(model_name)

        for seat, datapath in seat_data.items():

            metric = SEAT()
            dataloader = DataLoader(
                WeatDataset(datapath, tokenizer),
                batch_size=1,
                shuffle=False
            )

            for sample in dataloader:
                # nasty
                sample = [{key: val.to(device) for key, val in s.items()} for s in sample]

                with torch.no_grad():
                    x = model(sample[0])
                    y = model(sample[1])
                    a = model(sample[2])
                    b = model(sample[3])

                metric.update(x, y, a, b)

            seat_value = metric.compute()

            print(f'{seat}: {seat_value}')


if __name__ == "__main__":
    evaluate()
