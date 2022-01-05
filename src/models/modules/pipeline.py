from typing import Dict
from transformers import AutoModel, AutoModelForSequenceClassification
from src.utils.hidden_size import HIDDEN_SIZE
import torch
import torch.nn as nn


class Pipeline(nn.Module):

    AVAILABLE_DEBIASIG_MODES = ['sentence', 'token']

    def __init__(
        self,
        model_name: str,
        embedding_layer: str,
        debias_mode: str,
        hf_checkpoint: str = None,
        is_glue: bool = False,
    ) -> None:
        """Wrapper for 🤗's pipeline abstraction with custom embedding getter.

        Args:
            model_name: e.g.: `bert-base-uncased`
            embedding_layer: from where to get the embeddings?
                Available: CLS|first|last|all
                'CLS': sentence representation as embedding of the [CLS] token
                    (taken at the last hidden state)
                'first': first layer
                'last': last layer
                'all': all layers, stacked vertically
                'intermediate': layers index 1-4 inclusive
            debias_mode: sentence|token.
                'sentence': retruns embeddings of the whole sentence.
                'token': retruns words embeddings only, as indicated by 'keyword_mask'.
            hf_checkpoint: path to 🤗-compatibile checkpoint
            is_glue: if true, loads model "for sequence classification"
        """
        super().__init__()

        if debias_mode not in self.AVAILABLE_DEBIASIG_MODES:
            raise ValueError(f"Debiasing mode must be 'sentence' or 'token'. Given: {debias_mode}")

        self.model_name = model_name
        self.embedding_layer = embedding_layer
        self.return_word_embs = (debias_mode == 'token')
        self.is_glue = is_glue

        if not self.is_glue:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(hf_checkpoint)

    @property
    def dim(self):
        """Embedding dimension."""
        return HIDDEN_SIZE[self.model_name]

    def get_embeddings(self, outputs, embedding_layer) -> torch.tensor:
        if embedding_layer == 'CLS':
            return outputs.last_hidden_state[:, 0, :]

        if embedding_layer == 'first':
            return outputs.hidden_states[0]

        if embedding_layer == 'last':
            return outputs.hidden_states[-1]

        if embedding_layer == 'all':
            return torch.vstack(outputs.hidden_states)

        if embedding_layer == 'intermediate':
            return torch.vstack(outputs.hidden_states[1:5])

        raise NotImplementedError('embedding_layer must be first|last|all|intermediate|CLS.')

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
        # A quick workaround: in case of 'all'-layer embeddings, the effective
        # shape of `x` is (batch_sz * num_layers,  ...). Let's repeat mask
        # for each layer. NB: mask_repeats=1 in any other case.
        mask_repeats = x.shape[0] // mask.shape[0]
        mask = mask.repeat((mask_repeats, 1))

        # `x` contains only embeddings of these-sub tokens now.
        x_masked = self.apply_output_mask(x, mask)

        # The rest of token embeddings are zeroed. To compute the average,
        # we need number of non-zero embeddings - for each batch
        number_non_zer_embs = mask.sum(1, keepdim=True)

        if (number_non_zer_embs < 1).sum() > 0:
            raise ValueError("If provided, each row of the mask must be "
                             "non-zero. Please refine your data.")

        # Sum of sub-tokens for each batch-sentence
        subtoken_sum = x_masked.sum(1)

        # Eventually, we get the average of non-zero sub-tokens
        return subtoken_sum / number_non_zer_embs

    def forward_model(self, sentences):
        if self.is_glue:
            return self.model.bert(
                sentences['input_ids'],
                attention_mask=sentences['attention_mask'],
                output_hidden_states=True
            )
        return self.model(
            sentences['input_ids'],
            attention_mask=sentences['attention_mask'],
            output_hidden_states=True
        )

    def forward(
            self,
            sentences: Dict[str, torch.tensor],
            return_word_embs: bool = None,
            embedding_layer: str = None,
    ) -> torch.tensor:
        """Feed forward the model anbd extract embeddings.

        Args:
            sentences: tokenized sentence
            return_word_embs: if specified, shadows self.return_word_embs.
                If True, extracts word embeddings indicated by sentences['keyword_mask'].
            embedding_layer: if specified, shadows self.embedding_layer.
                One of: first|last|all|CLS.
        """
        outputs = self.forward_model(sentences)

        # Choose where to get embeddings from (first, last, all layers...)
        embedding_layer = self.embedding_layer if embedding_layer is None else embedding_layer

        return_word_embs = self.return_word_embs if return_word_embs is None else return_word_embs

        embeddings = self.get_embeddings(outputs, embedding_layer)

        if return_word_embs and embedding_layer != 'CLS':
            return self.get_word_embeddings(
                x=embeddings,
                mask=sentences['keyword_mask']
            )

        return embeddings
