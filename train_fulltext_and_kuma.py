#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os 
import argparse
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import datetime
import gc

date_time = str(datetime.date.today()) + "_" + ":".join(str(datetime.datetime.now()).split()[1].split(":")[:2])

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", 
    type = str, 
    help = "select dataset / task", 
    default = "sst", 
    choices = ["SST","IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr"]
)

parser.add_argument(
    "--data_dir", 
    type = str, 
    help = "directory of saved processed data", 
    default = "datasets/"
)

parser.add_argument(
    "--model_dir",   
    type = str, 
    help = "directory to save models", 
    default = "full_text_models/"
)

parser.add_argument(
    "--seed",   
    type = int, 
    help = "random seed for experiment"
)

parser.add_argument(
    '--evaluate_models', 
    help='test predictive performance in and out of domain', 
    action='store_true'
)

parser.add_argument(
    "--inherently_faithful", 
    type = str, 
    help = "select dataset / task", 
    default = None, 
    choices = [None, "kuma", "rl", "full_lstm"]
)

parser.add_argument(
    '--use_tasc', 
    help='for using the component by GChrys and Aletras 2021', 
    action='store_true'
)

user_args = vars(parser.parse_args())
user_args["importance_metric"] = None

log_dir = "experiment_logs/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" +  date_time + "/"
config_dir = "experiment_config/train_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" + date_time + "/"


os.makedirs(log_dir, exist_ok = True)
os.makedirs(config_dir, exist_ok = True)

import config.cfg

config.cfg.config_directory = config_dir

logging.basicConfig(
    filename= log_dir + "/out.log", 
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logging.info("Running on cuda : {}".format(torch.cuda.is_available()))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from src.common_code.initialiser import initial_preparations
import datetime

# creating unique config from user args and model_config.json file
args = initial_preparations(
    user_args, 
    stage = "train"
)

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")


if args["inherently_faithful"] is not None:
    
    from src.data_functions.dataholders import KUMA_RL_HOLDER as dataholder
    
else:
    
    from src.data_functions.dataholders import BERT_HOLDER as dataholder
    
from src.tRpipeline import train_and_save, test_predictive_performance, keep_best_model_

# training the models and evaluating their predictive performance
# on the full text length

data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"]
    )

## evaluating finetuned models
if args["evaluate_models"]:

    ## in domain evaluation
    test_stats = test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
        vocab_size = data.vocab_size
    )    

    del data
    gc.collect()

    ## ood evaluation ON OOD-DATASET 1
    data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        ood = True,
        ood_dataset_ = 1
    )

    test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
        ood = True,
        ood_dataset_ = 1,
        vocab_size = data.vocab_size
    )   

    ## ood evaluation ON OOD-DATASET 2
    data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        ood = True,
        ood_dataset_ = 2
    )

    test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
        ood = True,
        ood_dataset_ = 2,
        vocab_size = data.vocab_size
    )   

    ## shows which model performed best on dev F1 (in-domain)
    ## if keep_models = False then will remove the rest of the models to save space
    keep_best_model_(keep_models = False)

else:

    train_and_save(
        train_data_loader = data.train_loader, 
        dev_data_loader = data.dev_loader, 
        for_rationale = False, 
        output_dims = data.nu_of_labels,
        vocab_size = data.vocab_size
    )
