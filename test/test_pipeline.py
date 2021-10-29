import unittest
import torch

from src.models.modules.pipeline import Pipeline


# From project root dir:
# python -m unittest

class TestPipeline(unittest.TestCase):

    def setUp(self) -> None:

        self.model_name = 'distilbert-base-uncased'
        # self.tokenizer = Tokenizer(self.model_name)
        self.pipeline = Pipeline(self.model_name, 'last', 'sentence')

        # Common data

        # batch_size=2, max_seq_len=3, emb_dim=5
        self.outputs = torch.tensor([
            [
                [1, 1, 1],  # batch 0 - token0
                [2, 2, 2],  # batch 0 - token 1
            ],
            [
                [3, 3, 3],  # batch 1 - token 0
                [4, 4, 4]   # batch 1 - token 1
            ]
        ])

    def test_apply_output_mask(self):

        mask = torch.tensor([
            [1, 0],
            [0, 1],
        ])
        answer = torch.tensor([
            [
                [1, 1, 1],
                [0, 0, 0]],
            [
                [0, 0, 0],
                [4, 4, 4]
            ]
        ])

        result = self.pipeline.apply_output_mask(self.outputs, mask)
        self.assertTrue(torch.allclose(answer, result))

    def test_get_word_embeddings_failure(self):
        mask = torch.tensor([
            [1, 1],
            [0, 0],   # This must fail
        ])
        self.assertRaises(
            ValueError,
            self.pipeline.get_word_embeddings, self.outputs, mask
        )

    def test_get_word_embeddings(self):

        outs = torch.tensor([
            [                  # Values are already "masked" (for easy visuals)
                [1, 1, 1, 1],  # For this batch sum of embeddings is
                [2, 2, 2, 2],  # [3, 3, 3, 3] and there are two non-zero
                [0, 0, 0, 0]   # embeddigs -> [1.5, 1.5, 1.5, 1.5]
            ],
            [                  # Values are already "masked" (for easy visuals)
                [0, 0, 0, 0],  # For this batch sum of embeddings is
                [4, 4, 4, 4],  # [10, 10, 10, 10] and there are two non-zero
                [6, 6, 6, 6]   # embeddigs -> [5, 5, 5, 5]
            ],
        ], dtype=torch.float)

        mask = torch.tensor([
            [1, 1, 0],
            [0, 1, 1],
        ])

        answer = torch.tensor([
            [1.5, 1.5, 1.5, 1.5],
            [5.0, 5.0, 5.0, 5.0],
        ])

        result = self.pipeline.get_word_embeddings(x=outs, mask=mask)

        self.assertTrue(torch.allclose(answer, result))


if __name__ == '__main__':
    unittest.main()
