import unittest
import copy
import torch


from src.models.debiaser_pruned import DebiaserPruned


# From project root dir:
# python -m unittest

class TestPipeline(unittest.TestCase):

    def setUp(self):
        sparse_train_args = {
            'dense_pruning_method': 'sigmoied_threshold:1d_alt',
            'dense_block_rows': 1,
            'dense_block_cols': 1,
            'attention_pruning_method': 'sigmoied_threshold',
            'attention_block_rows': 64,
            'attention_block_cols': 768,
        }

        self.model_args = {
            'model_name': 'bert-base-uncased',
            'embedding_layer': 'last',
            'debias_mode': 'sentence',
            'learning_rate': 0,
            'weight_decay': 0,
            'adam_eps': 0,
            'warmup_steps': 0,
            'loss_alpha': 0,
            'loss_beta': 0,
            'sparse_train_args': sparse_train_args,
            'freeze_weights': True,
            'share_pruning_scores': True,
        }

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

    def test_values_only_pruning(self):
        model_args = copy.copy(self.model_args)

        model_args['share_pruning_scores'] = False
        model_args['prune_values_only'] = True

        model = DebiaserPruned(**model_args)
        model.train()

        optimizer = torch.optim.SGD(model.model_debias.parameters(), lr=0.001, momentum=0.9)
        gt = torch.rand(4, 128, 768)

        # simulate training loop
        for i in range(16):
            optimizer.zero_grad()

            sentences = {
                'input_ids':  torch.randint(0, 20000, (4, 128)),
                'attention_mask': torch.ones((4, 128), dtype=torch.long)
            }

            model.model_patcher.schedule_threshold(i, 10, True)
            y = model.model_debias(sentences)
            loss = (y - gt).pow(2).mean()
            loss.backward()
            optimizer.step()

        for name, p in model.model_debias.named_parameters():
            print(name, '\t', p.requires_grad)
            if 'mask_module' in name:
                if 'value' in name:
                    self.assertTrue(p.requires_grad)
                if 'query' in name or 'key' in name:
                    self.assertFalse(p.requires_grad)
                    self.assertTrue(torch.allclose(p.data, torch.tensor(420.0)))

    def test_everything(self):
        model = DebiaserPruned(**self.model_args)
        self.helper_freeze_weights(model)
        self.helper_score_share(model)
        self.helper_freeze_weights(model)
