#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import math 


def operation(scaled_embeddings, approach, dim = -1):
    
    """
    defines the operation over the scaled embedding
    """
    
    assert approach in ["sum-over", "max-pool", "mean-pool"]
    
    if approach == "sum-over":
        
        return scaled_embeddings.sum(dim)
    
    elif approach == "max-pool":
        
        return scaled_embeddings.max(dim)[0]
    
    else:
        
        return scaled_embeddings.mean(dim)
        
        
class lin(nn.Module):
    
    def __init__(self, vocab_size):
        
        super(lin, self).__init__()
        
        """
        Lin-TaSc where u is generated and multiplied with the embeddings
        to produce scaled non-contextualised embeddings
        
        operation over scaled embeddings to produce tasc scores s_i        
        """
        
        self.vocab_size = vocab_size
        
        self.u_param = nn.Parameter(torch.randn(self.vocab_size))
        stdv = 1. / math.sqrt(self.u_param.size(0))
        self.u_param.data.uniform_(-stdv, stdv)
        
    def forward(self, input_ids, embeddings):
        
        tasc_weights = self.u_param[input_ids]
                  
        scaled_embeddings = (tasc_weights.unsqueeze(-1) * embeddings)
        
        return operation(scaled_embeddings, "sum-over") ## hard coded as sum-over shown to perform better


class TanhAttention(nn.Module):
    def __init__(self, hidden_dim) :
        super(TanhAttention, self).__init__()
        
        self.attn1 =  nn.Linear(hidden_dim, 
                                hidden_dim // 2)
        stdv = 1. / math.sqrt(hidden_dim)
        self.attn1.weight.data.uniform_(-stdv, stdv)
        self.attn1.bias.data.fill_(0)

        self.attn2 = nn.Linear(hidden_dim // 2, 1, 
                        bias = False)
        stdv = 1. / math.sqrt(hidden_dim // 2) 
        self.attn2.weight.data.uniform_(-stdv, stdv)
        
  
    def forward(self, hidden, mask) :
          
     
        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        
        attn2.masked_fill_(mask.bool(), -float('inf'))

        weights = torch.softmax(attn2, dim = -1)
        
        return weights
