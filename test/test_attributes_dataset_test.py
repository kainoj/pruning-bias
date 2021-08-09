import unittest
import torch

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.dataset.attributes_dataset import AttributesWithSentecesDataset
from src.models.modules.mlm_pipeline import Pipeline


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

        self.model_name = 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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

    def test_apply_mask(self):
        """Check whether the pipeline applies the mask properly

        1. Wrap datset to dataloader
        2. For each batch, feed it to pipeline
        3. In outputs, find non-zeroed embeddings (their positions)
        4. And check whether these embeddings correnspond to tokens in the mask
        """
        pipeline = Pipeline(model_name=self.model_name, embeddings_from='last')

        dl = DataLoader(dataset=self.ds, batch_size=2, shuffle=False)

        # In total batches: 2 = 4 / 2 = len(ds) / batch_size
        self.assertEqual(len(dl), 2)

        # Answers[i] is all attributes in i-th sentence
        answers = ['tokenizer', 'tokenizer', 'pizza pizza', 'pizza']

        sample_no = 0

        for batch in dl:
            outputs = pipeline(batch)
            outputs = pipeline.apply_output_mask(
                outputs, mask=batch['attribute_mask']
            )

            # Output shape must have not been changed
            self.assertEqual(outputs.shape, (2, 128, 768)) # bsz, max_len, dim
    
            for sample in outputs:
                # sample is of shape (128, 768)
                # Sum elements of all 128 embeddings -> we should get 0 for
                # these embeddings tokens, that were masked properly
                non_zeros_mask = (sample.sum(1) != 0)

                original_tokens = self.ds[sample_no]['input_ids']
                
                only_attribuets_tokens = torch.masked_select(
                    original_tokens, # Encodings of sentences
                    non_zeros_mask   # Mask of attributes
                )
                decoded_str = self.tokenizer.decode(only_attribuets_tokens)
                self.assertEqual(decoded_str, answers[sample_no])

                sample_no += 1


if __name__ == '__main__':
    unittest.main()