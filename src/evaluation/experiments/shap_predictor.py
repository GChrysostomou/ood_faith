import torch
from torch import nn
import json
import config.cfg
from config.cfg import AttrDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class ShapleyModelWrapper(nn.Module):

    def __init__(self, model):

        super(ShapleyModelWrapper, self).__init__()

        self.model = model

    def forward(self, embeddings):        
        
        head_mask = [None] * self.model.wrapper.model.config.num_hidden_layers

        encoder_outputs = self.model.wrapper.model.encoder(
            embeddings,
            head_mask=head_mask,
            output_attentions=self.model.wrapper.model.config.output_attentions,
            output_hidden_states=self.model.wrapper.model.config.output_attentions,
            return_dict=self.model.wrapper.model.config.return_dict
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.model.wrapper.model.pooler(sequence_output) if self.model.wrapper.model.pooler is not None else None

        logits = self.model.output_layer(pooled_output)

        return logits