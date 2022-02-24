
import torch
import pandas as pd
import json 
import glob 
import os
import logging
from src.models.tasc import lin as tasc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

from src.models.bert import bert
from src.evaluation.experiments.rationale_extractor import extract_importance_, rationale_creator_, extract_lime_scores_, extract_shap_values_
from src.evaluation.experiments.erasure_tests import conduct_experiments_


import re

class constrastivator():

    """
    Class that contains method of rationale extraction as in:
        saliency scorer and thresholder approach
    Saves rationales in a csv file with their dedicated annotation_id 
    """

    def __init__(self, model_path, output_dims = 2, ood = False):
        
        """
        loads and holds a pretrained model
        """

        self.models = glob.glob(model_path + args["model_abbreviation"] + "*.pt")
        self.output_dims = output_dims
        self.ood = ood

        logging.info(f" *** there are {len(self.models)} models in :  {model_path}")

        if len(self.models) == 0:

            raise FileNotFoundError(
                f"*** no models in directory -> {model_path}"
            )

    def save_cls_(self, data, model = None):
    
        for model_name in self.models:
            
            if args.use_tasc:
            
                tasc_variant = tasc
                
                tasc_mech = tasc_variant(data.vocab_size)
                
            else:
                
                tasc_mech = None

            model = bert(
                output_dim = self.output_dims,
                tasc = tasc_mech
            )

            logging.info(f" *** loading model -> {model_name}")

            model.load_state_dict(torch.load(model_name, map_location=device))

            model.to(device)

            logging.info(f" *** succesfully loaded model -> {model_name}")
            
            ## preserves the tasc naming convention
            self.model_random_seed = re.sub("bert", "", model_name.split(".pt")[0].split("/")[-1]) 

            if self.ood:

                extract_importance_(
                    model = model, 
                    data_split_name = "test",
                    data = data.test_loader,
                    model_random_seed = self.model_random_seed,
                    ood = self.ood
                )

            else:

                extract_importance_(
                    model = model, 
                    data_split_name = "train",
                    data = data.train_loader,
                    model_random_seed = self.model_random_seed,
                    ood = self.ood
                )
           

        return

  