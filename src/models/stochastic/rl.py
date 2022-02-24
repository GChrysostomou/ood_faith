import numpy as np

import torch
from torch import nn
from torch.nn.functional import softplus
import json
import config.cfg
from config.cfg import AttrDict
import numpy as np
with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

from src.models.stochastic.util import get_z_stats
from src.models.stochastic.classifier import Classifier
from src.models.stochastic.nn.generator import IndependentGenerator
from src.models.stochastic.nn.generator import DependentGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RLModel(nn.Module):
    """
    Reimplementation of Lei et al. (2016). Rationalizing Neural Predictions
    for Stanford Sentiment.
    (Does classfication instead of regression.)
    Consists of:
    - Encoder that computes p(y | x, z)
    - Generator that computes p(z | x) independently or dependently with an RNN.
    """

    def __init__(self,
                 vocab:       object = None,
                 vocab_size:  int = 0,
                 emb_size:    int = 300,
                 hidden_size: int = 200,
                 output_dim:  int = 2,
                 dropout:     float = 0.1,
                 layer:       str = "lstm",
                 dependent_z: bool = False,
                 sparsity:    float = 0.0,
                 coherence:   float = 0.0,
                 tasc         = None
                 ):

        super(RLModel, self).__init__()

        assert tasc is None

        self.vocab = vocab
        self.sparsity = 0.001#(1-args.rationale_length)
        self.coherence = 0.001#coherence

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

        self.encoder = Classifier(
            embed=self.embedding, hidden_size=hidden_size, output_size=output_dim,
            dropout=dropout, layer=layer)

        if dependent_z:
            self.latent_model = DependentGenerator(
                embed=self.embedding, hidden_size=hidden_size,
                dropout=dropout, layer=layer)
        else:
            self.latent_model = IndependentGenerator(
                embed=self.embedding, hidden_size=hidden_size,
                dropout=dropout, layer=layer)

        self.criterion = nn.CrossEntropyLoss()

    def lagrange_parameters(self):
        return []

    @property
    def z(self):
        return self.latent_model.z

    @property
    def z_layer(self):
        return self.latent_model.z_layer

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)

    def forward(self, **inputs):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.
        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        self.mask = (inputs["input_ids"] != 0)[:,:max(inputs["lengths"])]
     
        lengths = self.mask.sum(1)
        
        z = self.latent_model(inputs["input_ids"], self.mask)
        logits = self.encoder(inputs["input_ids"], self.mask, z)
        
        return logits, None

    def get_loss(self, logits, targets, mask=None, **kwargs):
        """
        This computes the loss for the whole model.
        We stick to the variable names of the original code by Tao Lei
        as much as possible.
        :param logits:
        :param targets:
        :param mask:
        :param kwargs:
        :return:
        """

        optional = {}
        sparsity = self.sparsity
        coherence = self.coherence

        loss_vec = self.criterion(logits, targets)  # [B]

        # main MSE loss for p(y | x,z)
        loss = loss_vec.mean()        # [1]
        optional["ce"] = loss.item()  # [1]

        # compute generator loss
        z = self.latent_model.z.squeeze(1).squeeze(-1)  # [B, T]

        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, self.mask)
            optional["p1"] = num_1 / float(total)

        # get P(z = 0 | x) and P(z = 1 | x)
        if len(self.latent_model.z_dists) == 1:  # independent z
            m = self.latent_model.z_dists[0]
            logp_z0 = m.log_prob(0.).squeeze(2)  # [B,T], log P(z = 0 | x)
            logp_z1 = m.log_prob(1.).squeeze(2)  # [B,T], log P(z = 1 | x)
        else:  # for dependent z case, first stack all log probs
            logp_z0 = torch.stack(
                [m.log_prob(0.) for m in self.latent_model.z_dists], 1).squeeze(2)
            logp_z1 = torch.stack(
                [m.log_prob(1.) for m in self.latent_model.z_dists], 1).squeeze(2)

        # compute log p(z|x) for each case (z==0 and z==1) and mask
        logpz = torch.where(z == 0, logp_z0, logp_z1)
        logpz = torch.where(self.mask, logpz, logpz.new_zeros([1]))

        # sparsity regularization
        zsum = z.sum(1)  # [B]
        zdiff = z[:, 1:] - z[:, :-1]
        zdiff = zdiff.abs().sum(1)  # [B]

        zsum_cost = sparsity * zsum.mean(0)
        optional["zsum_cost"] = zsum_cost.item()

        zdiff_cost = coherence * zdiff.mean(0)
        optional["zdiff_cost"] = zdiff_cost.mean().item()

        sparsity_cost = zsum_cost + zdiff_cost
        optional["sparsity_cost"] = sparsity_cost.item()

        cost_vec = loss_vec.detach() + zsum * sparsity + zdiff * coherence
        cost_logpz = (cost_vec * logpz.sum(1)).mean(0)  # cost_vec is neg reward

        obj = cost_vec.mean()  # MSE with regularizers = neg reward
        optional["obj"] = obj.item()

        # generator cost
        optional["cost_g"] = cost_logpz.item()

        # encoder cost
        optional["cost_e"] = loss.item()

        return loss + cost_logpz, optional