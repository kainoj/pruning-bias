from utils.sentence_embbeder import SentenceEmbedder
from metrics.seat import SEAT6, SEAT7, SEAT8


if __name__ == "__main__":

    model_name = 'distilbert-base-uncased'
    
    embedder = SentenceEmbedder(model_name)

    seat_metrics = {"SEAT6": SEAT6(), "SEAT7": SEAT7(), "SEAT8": SEAT8()}

    for name in seat_metrics:
        value = seat_metrics[name](embedder)
        print(f"{name}: {value}")
