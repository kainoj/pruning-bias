import unittest
import torch

from torch.utils.data import DataLoader
from src.dataset.keywords_dataset import SentencesWithKeywordsDataset
from src.models.modules.pipeline import Pipeline
from src.models.modules.tokenizer import Tokenizer


# From project root dir:
# python -m unittest

class AttributesDatasetTest(unittest.TestCase):

    def setUp(self) -> None:

        sentences = [
            "i like tokenizer",
            "tokenizer with pineapple",
            "i like pizza pizza",
            "pizza with pineapple",
        ]

        attributes = [
            ["tokenizer"], ["tokenizer"], ["pizza"], ["pizza"]
        ]

        self.model_name = 'distilbert-base-uncased'
        self.tokenizer = Tokenizer(self.model_name)
        self.pipeline = Pipeline(model_name=self.model_name, embedding_layer='last')

        self.ds = SentencesWithKeywordsDataset(
            sentences=sentences,
            keywords=attributes,
            tokenizer=self.tokenizer,
        )

    def test_ds_len(self):
        # 4 = total #sentences
        self.assertEqual(len(self.ds), 4)

    def test_get_item(self):
        """Get i-th item, extract tokens from the mask and check whether
        they decode to attributes."""
        # Answers[i] is all attributes in i-th sentence
        answers = ['tokenizer', 'tokenizer', 'pizza pizza', 'pizza']

        for ans, data in zip(answers, self.ds):

            # Extract tokens only for attributes
            only_attributes_tokens = torch.masked_select(
                data['input_ids'],             # Encodings of sentences
                data['keyword_mask'].bool()  # Mask of attributes
            )

            decoded_str = self.tokenizer.decode(only_attributes_tokens)

            self.assertEqual(decoded_str, ans)

    def test_apply_mask(self):
        """Check whether the pipeline applies the mask properly

        1. Wrap datset to dataloader
        2. For each batch, feed it to pipeline
        3. In outputs, find non-zeroed embeddings (their positions)
        4. And check whether these embeddings correnspond to tokens in the mask
        """
        dl = DataLoader(dataset=self.ds, batch_size=2, shuffle=False)

        # In total batches: 2 = 4 / 2 = len(ds) / batch_size
        self.assertEqual(len(dl), 2)

        # Answers[i] is all attributes in i-th sentence
        answers = ['tokenizer', 'tokenizer', 'pizza pizza', 'pizza']

        sample_no = 0

        for batch in dl:
            outputs = self.pipeline(batch)
            outputs = self.pipeline.apply_output_mask(
                outputs, mask=batch['keyword_mask']
            )

            # Output shape must have not been changed
            self.assertEqual(outputs.shape, (2, 128, 768))  # bsz, max_len, dim

            for sample in outputs:
                # sample is of shape (128, 768)
                # Sum elements of all 128 embeddings -> we should get 0 for
                # these embeddings tokens, that were masked properly
                non_zeros_mask = (sample.sum(1) != 0)

                original_tokens = self.ds[sample_no]['input_ids']

                only_attributes_tokens = torch.masked_select(
                    original_tokens,  # Encodings of sentences
                    non_zeros_mask    # Mask of attributes
                )
                decoded_str = self.tokenizer.decode(only_attributes_tokens)
                self.assertEqual(decoded_str, answers[sample_no])

                sample_no += 1

    def test_get_word_embeddings(self):

        outs = torch.tensor([
            [
                [1, 1, 1, 1],  # For this batch sum of embeddings is
                [2, 2, 2, 2],  # [3, 3, 3, 3] and there are two non-zero
                [0, 0, 0, 0]   # embeddigs -> [1.5, 1.5, 1.5, 1.5]
            ],
            [
                [0, 0, 0, 0],  # For this batch sum of embeddings is
                [4, 4, 4, 4],  # [10, 10, 10, 10] and there are two non-zero
                [6, 6, 6, 6]   # embeddigs -> [5, 5, 5, 5]
            ]
        ], dtype=torch.float)

        mask = torch.tensor([
            [1, 1, 0],
            [0, 1, 1]
        ])

        answer = torch.tensor([
            [1.5, 1.5, 1.5, 1.5],
            [5.0, 5.0, 5.0, 5.0]
        ])

        result = self.pipeline.get_word_embeddings(x=outs, mask=mask)

        self.assertTrue(torch.allclose(answer, result))


if __name__ == '__main__':
    unittest.main()
