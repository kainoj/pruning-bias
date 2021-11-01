import hydra
import torch
import csv

from omegaconf import DictConfig
from pathlib import Path

from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer
from src.dataset.weat_dataset import WeatDataset
from src.metrics.seat import SEAT


def dict_to_device(dictornary, device):
    return {key: val.to(device) for key, val in dictornary.items()}


def evaluate(model_name, device, seat_data, data_root):

    model = Pipeline(model_name, embedding_layer='CLS', debias_mode='sentence').to(device)
    tokenizer = Tokenizer(model_name)

    results = {}

    print(f"Evaluating: '{model_name}'")

    for seat, datapath in seat_data.items():

        metric = SEAT()

        seat_data = WeatDataset(data_root / datapath, tokenizer).get_all_items()
        seat_data = {subset: dict_to_device(data, device) for subset, data in seat_data.items()}

        with torch.no_grad():
            x = model(seat_data['target_x'])
            y = model(seat_data['target_y'])
            a = model(seat_data['attribute_a'])
            b = model(seat_data['attribute_b'])

        metric.update(x, y, a, b)
        seat_value = metric.compute()

        print(f'{seat}:\t{seat_value}')

        results[seat] = seat_value.item()

    results['model'] = model_name
    return results


def to_csv(scores_dict, filename):

    should_write_header = not Path(filename).exists()

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=scores_dict.keys())
        if should_write_header:
            writer.writeheader()
        writer.writerow(scores_dict)


@hydra.main(config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:

    model_name = cfg.model_name
    device = cfg.device
    seat_data = cfg.seat_data
    data_root = cfg.data_root if cfg.data_root else Path(hydra.utils.get_original_cwd())

    results = evaluate(model_name, device, seat_data, data_root)

    if Path(model_name).is_dir():
        seat_outfile = Path(model_name) / 'seat.csv'
        to_csv(results, seat_outfile)

        print(f'Saved scores at {seat_outfile}')


if __name__ == "__main__":
    main()
