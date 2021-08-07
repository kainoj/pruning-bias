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

    def forward(self, sentences: List[str]) -> torch.tensor:
        # TODO: change sentences type to tokenized stuff
        outputs = self.model(
            sentences['input_ids'],
            attention_mask=sentences['attention_mask'], 
            output_hidden_states=True
        )

        if self.embeddings_from == 'CLS':
            return outputs.last_hidden_state[:, 0, :]
        
        if self.embeddings_from == 'first':
            return outputs.hidden_states[0]

        if self.embeddings_from == 'last':
            return outputs.hidden_states[-1]
        
        if self.embeddings_from == 'all':
            return outputs.hidden_states  #  concat maybe?
        
        raise NotImplementedError()

