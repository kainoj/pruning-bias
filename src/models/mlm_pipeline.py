from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

class SentenceEmbedder(nn.Module):

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
        self.embeddings_from = embeddings_from
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # take care whether it really sets model in train/eval/gpt etc 
        self.add_module(f'custom-{model_name}', self.model)

    def forward(self, sentences: List[str]) -> torch.tensor:

        # Tokenizer does automatically prepend [CLS] to sentences.
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)

        outputs = self.model(**inputs, output_hidden_states=True)

        
        if self.embeddings_from == 'CLS':
            return outputs.last_hidden_state[:, 0, :]
        
        if self.embeddings_from == 'first':
            return outputs.hidden_states[0]

        if self.embeddings_from == 'last':
            return outputs.hidden_states[-1]
        
        if self.embeddings_from == 'all':
            return outputs.hidden_states  #  concat maybe?
        
        raise NotImplementedError()

