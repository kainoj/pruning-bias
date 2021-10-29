import unittest
import regex as re
import torch

from src.models.modules.tokenizer import Tokenizer
from src.dataset.utils import get_keyword, get_keyword_mask


# From project root dir:
# python -m unittest

class DatasetUtilsTest(unittest.TestCase):

    def setUp(self) -> None:

        self.model_name = 'distilbert-base-uncased'
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # noqa E501

    def test_get_keyword(self):

        # Word "pizza" is a male attr and appears in the sentence
        result = get_keyword(
            example={'text': "I like pizza."},
            male_attr={"pizza"},
            female_attr={"token"},
            stereo_trgt={"pineapple"},
            pattern=self.pattern)

        self.assertEqual(result['keywords'], ['pizza'])
        self.assertEqual(result['type'], 'male')

        # Sentence contains both male and female attr -> nothing gets extracted
        result = get_keyword(
            example={'text': "I like pizza token."},
            male_attr={"pizza"},
            female_attr={"token"},
            stereo_trgt={"pineapple"},
            pattern=self.pattern)

        self.assertEqual(result['keywords'], [])
        self.assertEqual(result['type'], 'none')

    def test_get_keyword_mask(self):
        tokenizer = Tokenizer(self.model_name)

        sentence = "I like pizza with pizza apple watch."

        test_data = [
            {
                'keywords': ['pizza'],
                'ans': 'pizza pizza'
            },
            {
                'keywords': ['pizza', 'apple'],
                'ans': 'pizza pizza apple',
            },
            {
                'keywords': ["glass"],
                'ans': '',
            },
            {
                'keywords': [],
                'ans': '',
            }
        ]

        for data in test_data:
            example = tokenizer(sentence)
            example['keywords'] = data['keywords']

            result = get_keyword_mask(example, tokenizer)

            # Extract tokens only for attributes
            only_attributes_tokens = torch.masked_select(
                torch.tensor(result['input_ids']),   # Encodings of sentences
                result['keyword_mask'].bool()        # Mask of attributes
            )
            self.assertEqual(
                tokenizer.decode(only_attributes_tokens),
                data['ans']
            )


if __name__ == '__main__':
    unittest.main()
