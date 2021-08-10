from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn


class Pipeline(nn.Module):

    def __init__(self, model_name: str, embeddings_from: str='last') -> None:
        """Wrapper for 🤗's pipeline abstraction with custom embedding getter.

        Args:
            model_name: e.g.: `distilbert-base-uncased`
            embeddings_from: from where to get the embeddings?
                Available: CLS|first|last|all 
                'CLS': sentence representation as embedding of [CLS] token
                 (taken at the last hidden state)
                'first':
                'last':
                'all': 
        """
        super().__init__()

        self.embeddings_from = embeddings_from
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # take care whether it really sets model in train/eval/gpt etc 
        self.add_module(f'custom-{model_name}', self.model)

    def get_embeddings(self, outputs) -> torch.tensor:
        if self.embeddings_from == 'CLS':
            return outputs.last_hidden_state[:, 0, :]
        
        if self.embeddings_from == 'first':
            return outputs.hidden_states[0]

        if self.embeddings_from == 'last':
            return outputs.hidden_states[-1]
        
        if self.embeddings_from == 'all':
            return outputs.hidden_states  #  concat maybe?
        
        raise NotImplementedError()

    def apply_output_mask(
            self, x: torch.tensor, mask: torch.tensor
        ) -> torch.tensor:
        """Extract specific tokens from `x`, defined by a `mask`.

        Used, for example, to extract embeddings of specific tokens.

        Args:
            x: inputs of shape (batch_sz, max_seq_len, emb_dim)
            mask: binary mask of shape (batch_sz, max_seq_len)

        Return:
            Tensor `y`, such that embedding y[i, j] is zeroed iff mask[i, j]==0
        """
        # Fist two dimensions must agree, so we can
        # broadcast mask values ove the entire embeddings
        assert x.shape[:2] == mask.shape

        mask_size = (x.shape[0], x.shape[1], 1)

        y = x * mask.reshape(mask_size)
        return y

    def get_word_embeddings(
            self, x: torch.tensor, mask: torch.tensor
        ) -> torch.tensor:
        """Extracts word embeddings from embedded sentence, based on the mask.
        
        'If a word is split into multiple sub-tokens, we compute the
        contextualized embedding of the word by averaging the contextualized
        embeddings of its constituent sub-tokens' (The paper, Section 3).

        Args:
            x: embeddings of shape (batch_sz, max_seq_len, emb_dim)
            mask: binary mask indicating postitions of sub-tokens

        Return:  a word embedding being an avg of it's sub-tokens.
            Shape: (batch_sz, emb_dim).
        """
        # `x` contains only embeddings of these-sub tokens now.
        x_masked = self.apply_output_mask(x, mask)

        # The rest of token embeddings are zeroed. To compute the average,
        # we need number of non-zero embeddings - for each batch
        number_non_zer_embs = mask.sum(1, keepdim=True)

        # Sum of sub-tokens for each batch-sentence
        subtoken_sum = x_masked.sum(1) 

        # Eventually, we get the average of non-zero sub-tokens
        return subtoken_sum / number_non_zer_embs

    def forward(self, sentences, return_word_embs=False) -> torch.tensor:
        # TODO: change sentences type to tokenized stuff
        outputs = self.model(
            sentences['input_ids'],
            attention_mask=sentences['attention_mask'], 
            output_hidden_states=True
        )

        # Choose where to get embeddings from (first, last, all layers...)
        embeddings = self.get_embeddings(outputs)
    
        if return_word_embs:
            return self.get_word_embeddings(
                x=embeddings,
                mask=sentences['attribute_mask']
            )
        return embeddings
