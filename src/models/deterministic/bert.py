import torch
import torch.nn as nn
import math 
from transformers import AutoModel, AutoConfig
import json 
from src.models.deterministic.bert_components import BertModelWrapper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
            args = AttrDict(json.load(f))

class BertClassifier(nn.Module):
    def __init__(self, output_dim = 2, dropout=0.1, tasc = None):
        
        super(BertClassifier, self).__init__()

        self.output_dim = output_dim        
        self.dropout = dropout

        self.bert_config = AutoConfig.from_pretrained(
            args["model"], 
            output_attentions = True
        )   
        
        self.wrapper = BertModelWrapper(
            AutoModel.from_pretrained(
                args["model"], 
                config=self.bert_config
            )
        )

        self.tasc_mech = tasc
  
        self.dropout = nn.Dropout(p = self.dropout)

        self.output_layer = nn.Linear(self.wrapper.model.config.hidden_size, self.output_dim)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, **inputs):

        if "ig" not in inputs: inputs["ig"] = int(1)

        _, pooled_output, attention_weights = self.wrapper(
            inputs["input_ids"], 
            attention_mask = inputs["attention_mask"],
            token_type_ids = inputs["token_type_ids"],
            ig = inputs["ig"],
            tasc_mech = self.tasc_mech 
        )

        # to retain gradients
        self.weights_or = attention_weights[-1]

        if inputs["retain_gradient"]:
            
            self.wrapper.word_embeds.retain_grad()
            self.weights_or.retain_grad()

        # attention weight indexing same as Learning to Faithfully Rationalise by Construction (FRESH)
        self.weights = self.weights_or[:, :, 0, :].mean(1)	
        
        logits = self.output_layer(pooled_output)
        
        return logits, self.weights

    def integrated_grads(self, original_grad, original_pred, steps = 10, **inputs):

        grad_list = [original_grad]
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            
            self.eval()
            self.zero_grad()

            inputs["ig"] = x
            
            pred, _ = self.forward(**inputs)

            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]

            pred[rows, original_pred[1]].sum().backward()

            #embedding gradients
            embed_grad = self.wrapper.model.embeddings.word_embeddings.weight.grad
            g = embed_grad[inputs["input_ids"].long()]

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.wrapper.model.embeddings.word_embeddings.weight[inputs["input_ids"].long()]

        ig = torch.norm(attributions* em, dim = -1)
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))

        return ig

