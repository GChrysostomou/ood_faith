import torch
from torch import nn
import json
import config.cfg
from config.cfg import AttrDict
import numpy as np
with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

from src.models.stochastic.util import get_encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = ["vanilla"]


class LSTMClassifier(nn.Module):
    """
    An LSTM Classifier with pre-trained embeddings frozen
    """
    def __init__(self,
                 emb_size:       int = 300,
                 hidden_size:    int = 200,
                 output_dim:     int = 2,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 tasc:           bool = None
                 ):

        super(LSTMClassifier, self).__init__()

        if tasc is not None:
            
            raise NotImplementedError
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_dim

        embedding = np.load(f"{args.data_dir}{args.embed_model}_embeds.npy") 
        self.vocab_size = embedding.shape[0]
        
        ## load pre-trained embeddings
        self.embedding = nn.Embedding(self.vocab_size, emb_size, padding_idx=0)
        
        self.embedding.load_state_dict(
            {"weight":torch.tensor(embedding).float()}
        )
            
        ## assertion that pretrained embeddings where loaded correctly
        assert torch.eq(self.embedding.weight, torch.tensor(embedding).float()).sum() == self.embedding.weight.numel()        
        
        # we do not train pretrained embeddings
        self.embedding.weight.requires_grad = False

        self.embedding = nn.Sequential(self.embedding, nn.Dropout(p=dropout))
        self.enc_layer = get_encoder(layer, emb_size, hidden_size)


        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size * 2, self.output_size)
        )

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, **inputs):
        
        self.mask = (inputs["input_ids"] != 0)[:,:max(inputs["lengths"])]

        emb = self.embedding(inputs["input_ids"])  # [B, T, E]

        # encode the sentence
        _, final = self.enc_layer(emb, self.mask,  self.mask.sum(1)) # [B,H]

        # predict sentiment from final state(s)
        logits = self.output_layer(final) 

        return logits, None

    def get_loss(self, logits, targets):

        optional = {}

        return self.criterion(logits, targets), optional  # [B]