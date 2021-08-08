import unittest
import torch

from transformers import AutoTokenizer
from src.dataset.attributes_dataset import AttributesWithSentecesDataset


# From project root dir:
# python -m unittest

class AttributesDatasetTest(unittest.TestCase):

    def setUp(self) -> None:
        attributes = [
            "tokenizer",  # "tokenizer" gets tokenized to 2 tokens
            "pizza"       # "pizza" gets tokenized to 1 token
        ]
        sentences_of_attributes = [
            ["i like tokenizer", "tokenizer with pineapple"],
            ["i like pizza pizza", "pizza with pineapple"]
        ]

        model_name = 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.ds = AttributesWithSentecesDataset(
            attributes=attributes,
            sentences_of_attributes=sentences_of_attributes,
            tokenizer=self.tokenizer
        )

    def test_ds_len(self):
        # 4 = total #sentences
        self.assertEqual(len(self.ds), 4) 

    def test_get_item(self):
        # Answers[i] is all attributes in i-th sentence
        answers = ['tokenizer', 'tokenizer', 'pizza pizza', 'pizza']

        for ans, data in zip(answers, self.ds):
           
            # Extract tokens only for attributes
            only_attribuets_tokens = torch.masked_select(
                data['input_ids'],             # Encodings of sentences
                data['attribute_mask'].bool() # Mask of attributes
            )

            decoded_str = self.tokenizer.decode(only_attribuets_tokens)

            self.assertEqual(decoded_str, ans)


if __name__ == '__main__':
    unittest.main()