import torch
import torch.nn as nn
import json


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class aDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def bert_embeddings(bert_model, 
                    input_ids, 
                    position_ids = None, 
                    token_type_ids = None):

    """
    forward pass for the bert embeddings
    """

    if input_ids is not None:

        input_shape = input_ids.size()

    seq_length = input_shape[1]

    if position_ids is None:

        position_ids = torch.arange(512).expand((1, -1)).to(device)
        position_ids = position_ids[:, :seq_length]

    if token_type_ids is None:
    
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

    embed = bert_model.embeddings.word_embeddings(input_ids)
    position_embeddings = bert_model.embeddings.position_embeddings(position_ids)
    token_type_embeddings = bert_model.embeddings.token_type_embeddings(token_type_ids)

    embeddings = embed + position_embeddings + token_type_embeddings
    embeddings = bert_model.embeddings.LayerNorm(embeddings)
    embeddings = bert_model.embeddings.dropout(embeddings)

    return embeddings, embed


class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
    
        super(BertModelWrapper, self).__init__()

        """
        BERT model wrapper
        """

        self.model = model
        
    def forward(self, input_ids, attention_mask, token_type_ids, ig = int(1), tasc_mech = None):        
        
        embeddings, self.word_embeds = bert_embeddings(
            self.model, 
            input_ids = input_ids, 
            position_ids = None, 
            token_type_ids = token_type_ids
        )

        ## if its for evaluation we need it to be a fraction
        if type(ig) == int:

            assert ig >= 0. and ig <= int(1), "IG ratio cannot be out of the range 0-1"

        else:
            
            ## else we need it to match the embeddings size for the KUMA mask
            ## therefore in this case ig is actually z from the KUMA model
            assert ig.size(0) == embeddings.size(0), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(1) == embeddings.size(1), "Mis-match in dimensions of mask and embeddings"
            assert ig.size(2) == 1, "Rationale mask should be of size 1 in final dimension"

            ig = ig.float()
  
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.model.config.num_hidden_layers

        encoder_outputs = self.model.encoder(
            embeddings * ig,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_attentions,
            return_dict=self.model.config.return_dict
        )

        sequence_output = encoder_outputs[0]

        if tasc_mech:

            tasc_weights = tasc_mech(
                input_ids = input_ids, 
                embeddings = self.word_embeds
            )

            pad_id = 0
            mask = (input_ids == pad_id)
            
            tasc_weights.masked_fill_(mask.bool(), 0.)

            sequence_output = sequence_output * tasc_weights.unsqueeze(-1)

        attentions = encoder_outputs[2]
        pooled_output = self.model.pooler(sequence_output) if self.model.pooler is not None else None

        return sequence_output, pooled_output, attentions
