"""What is dimensionality of the encoder layers and the pooler layer of your model?

Different models call this differently. For example vanilla BERT calls
is a 'hidden_size', whereas DistilBERT: 'dim'.

Based on https://huggingface.co/transformers/pretrained_models.html
"""

HIDDEN_SIZE = {
    'distilbert-base-uncased': 768,
    'bert-base-uncased': 768,
    'roberta-base': 768,
    'albert-base-v2': 768,
    'google/electra-small-discriminator': 256,  # https://github.com/google-research/electra
}
