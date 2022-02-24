#!/usr/bin/env python

import torch
from torch import nn
import json
import config.cfg
from config.cfg import AttrDict
import numpy as np
with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

from src.models.stochastic.util import get_z_stats
from src.models.stochastic.classifier import Classifier
from src.models.stochastic.latent import \
    DependentLatentModel, IndependentLatentModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = ["LatentRationaleModel"]


class LatentRationaleModel(nn.Module):
    """
    Latent Rationale
    Categorical output version (for SST)
    Consists of:
    p(y | x, z)     observation model / classifier
    p(z | x)        latent model
    """
    def __init__(self,
                 emb_size:       int = 300,
                 hidden_size:    int = 200,
                 output_dim:     int = 2,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 dependent_z:    bool = False,
                 z_rnn_size:     int = 30,
                 lasso:          float = 0.0,
                 lambda_init:    float = 1e-3,
                 lagrange_lr:    float = 0.01,
                 lagrange_alpha: float = 0.99, # was 0.99
                 tasc:           bool = None
                 ):

        super(LatentRationaleModel, self).__init__()

        if tasc is not None:
            
            raise NotImplementedError
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_dim
        self.selection = args.rationale_length#selection#args.rationale_length
        self.lasso = lasso

        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

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

        self.classifier = Classifier(
            embed=self.embedding, hidden_size=hidden_size, output_size=self.output_size,
            dropout=dropout, layer=layer)

        if self.dependent_z:
            self.latent_model = DependentLatentModel(
                embed=self.embedding, hidden_size=hidden_size,
                dropout=dropout, layer=layer)
        else:
            self.latent_model = IndependentLatentModel(
                embed=self.embedding, hidden_size=hidden_size,
                dropout=dropout, layer=layer)

        self.criterion = nn.CrossEntropyLoss()

        # lagrange
        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average

    @property
    def z(self):
        return self.latent_model.z

    @property
    def z_layer(self):
        return self.latent_model.z_layer

    @property
    def z_dists(self):
        return self.latent_model.z_dists

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


        y = self.classifier(inputs["input_ids"], self.mask, z)

        return y, None ## returning None for compatibility when using bert

    def get_loss(self, logits, targets, mask=None, **kwargs):

        optional = {}
        selection = self.selection
        lasso = self.lasso

        loss_vec = self.criterion(logits, targets)  # [B]

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item()  # [1]

        batch_size = self.mask.size(0)
        lengths = self.mask.sum(1).float()  # [B]

        # z = self.generator.z.squeeze()
        z_dists = self.latent_model.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)
        pdf0 = torch.where(self.mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        pdf_nonzero = 1. - pdf0  # [B, T]
        pdf_nonzero = torch.where(self.mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        c0_hat = (l0 - selection)

        # moving average of the constraint
        self.c0_ma = self.lagrange_alpha * self.c0_ma + \
            (1 - self.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        loss = ce + self.lambda0.detach() * c0

        if lasso > 0.:
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            zt_zero = pdf0[:, :-1]
            ztp1_nonzero = pdf_nonzero[:, 1:]

            # cost z_t = non-zero, z_{t+1} = zero
            zt_nonzero = pdf_nonzero[:, :-1]
            ztp1_zero = pdf0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            lasso_cost = lasso_cost * self.mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange coherence dissatisfaction (batch average)
            target1 = lasso

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = (lasso_cost - target1)

            # update moving average
            self.c1_ma = self.lagrange_alpha * self.c1_ma + \
                (1 - self.lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(
                self.lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["target1"] = lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.lambda1.item()
                optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

            loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, self.mask)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional