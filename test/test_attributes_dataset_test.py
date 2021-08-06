import unittest
import torch

from transformers import AutoTokenizer
from src.dataset.attributes_dataset import AttributesDataset


# From project root dir:
# python -m unittest

class AttributesDatasetTest(unittest.TestCase):

    def test_get_attributes(self):
        sentences = None
        targets_in_sentences = None
        attr2sent = {
            # "tokenizer" gets tokenized to 2 tokens
            "tokenizer": [
                "i like tokenizer",
                "tokenizer with pineapple"
            ],
            # "pizza" gets tokenized to 1 token
            "pizza": [
                "i like pizza pizza",
                "pizza with pineapple"
            ]
        }

        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        ds = AttributesDataset(
            sentences=sentences,
            targets_in_sentences=targets_in_sentences,
            attr2sent=attr2sent,
            tokenizer=tokenizer
        )
        
        # We will check only for total number of occurrences of attributes
        answers = [
            'tokenizer tokenizer',
            'pizza pizza pizza'
        ]

        results = ds.get_attributes_with_sentences()

        for ans, data in zip(answers, results):
           
            # Extract tokens only for attributes
            only_attribuets_tokens = torch.masked_select(
                data['input_ids'],             # Encodings of sentences
                data['attributes_mask'].bool() # Mask of attributes
            )

            decoded_str = tokenizer.decode(only_attribuets_tokens)
            
            self.assertEqual(decoded_str, ans)

if __name__ == '__main__':
    unittest.main()