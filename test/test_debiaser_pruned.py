import unittest

from src.models.debiaser_pruned import DebiaserPruned


# From project root dir:
# python -m unittest

class TestPipeline(unittest.TestCase):

    def helper_freeze_weights(self, model):
        for name, p in model.model_debias.named_parameters():
            if 'mask_scores' in name:
                self.assertTrue(p.requires_grad)
            else:
                self.assertFalse(p.requires_grad)

    def helper_score_share(self, model):
        # For each layer
        for layer in model.model_debias.model.encoder.layer:
            per_layer_ptrs = []
            # Check whether Q, K, V scores point to the same mem
            for name, param in layer.named_parameters():
                if 'self' in name and 'mask_scores' in name:
                    per_layer_ptrs.append(param.data_ptr())
            self.assertEqual(len(set(per_layer_ptrs)), 1)

    def test_everything(self):

        sparse_train_args = {
            'dense_pruning_method': 'sigmoied_threshold:1d_alt',
            'dense_block_rows': 1,
            'dense_block_cols': 1,
            'attention_pruning_method': 'sigmoied_threshold',
            'attention_block_rows': 64,
            'attention_block_cols': 768,
        }

        model = DebiaserPruned(
            'bert-base-uncased',
            embedding_layer='last',
            debias_mode='sentence',
            learning_rate=0,
            weight_decay=0,
            adam_eps=0,
            warmup_steps=0,
            loss_alpha=0,
            loss_beta=0,
            sparse_train_args=sparse_train_args,
            freeze_weights=True,
            share_pruning_scores=True,
        )

        self.helper_freeze_weights(model)
        self.helper_score_share(model)
        self.helper_freeze_weights(model)
