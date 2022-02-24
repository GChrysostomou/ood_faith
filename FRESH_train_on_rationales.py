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
    "--extracted_rationale_dir", 
    type = str, 
    help = "directory of saved processed data", 
    default = "datasets/"
)

parser.add_argument(
    "--rationale_model_dir",   
    type = str, 
    help = "directory to save models", 
    default = "rationale_models/"
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
    "--importance_metric", 
    type = str, 
    help = "importance metric for ra.ext.", 
    default = "attention", 
    choices = ["attention", "gradients", "scaled attention", "ig", "deeplift"]
)

parser.add_argument(
    "--thresholder", 
    type = str, 
    help = "thresholder for extracting rationales", 
    default = "topk",
    choices = ["contigious", "topk"]
)

parser.add_argument(
    '--train_on_ood', 
    help='train on out of domain an inherently faithful classifier', 
    action='store_true'
)

parser.add_argument(
    '--use_tasc', 
    help='for using the component by GChrys and Aletras 2021', 
    action='store_true'
)

parser.add_argument(
    "--inherently_faithful", 
    type = str, 
    help = "select dataset / task", 
    default = None, 
    choices = [None]
)



user_args = vars(parser.parse_args())

log_dir = "experiment_logs/train_on_RAT_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" +  date_time + "/"
config_dir = "experiment_config/train_on_RAT_" + user_args["dataset"] + "_seed-" + str(user_args["seed"]) + "_" + date_time + "/"

assert user_args["inherently_faithful"] is None, (
    """
    Cannot use inherently faithful with FRESH
    """
)

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

# creating unique config from stage_config.json file and model_config.json file
args = initial_preparations(user_args, stage = "retrain")

logging.info("config  : \n ----------------------")
[logging.info(k + " : " + str(v)) for k,v in args.items()]
logging.info("\n ----------------------")



from src.data_functions.dataholders import BERT_HOLDER as dataholder
from src.tRpipeline import train_and_save, test_predictive_performance, keep_best_model_

# training the models and evaluating their predictive performance
# on the full text length
data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        stage = "train",
        for_rationale = True
    )

## evaluating finetuned models
if args["evaluate_models"]:

    ## in domain evaluation
    test_stats = test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = True, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
        vocab_size = data.vocab_size
    )    

    del data
    gc.collect()

    # ood evaluation DATASET 1
    data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        ood = True,
        ood_dataset_ = 1,
        stage = "train",
        for_rationale = True
    )

    test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = True, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
        ood = True,
        ood_dataset_ = 1,
        vocab_size = data.vocab_size
    )   

    del data
    gc.collect()

    # ood evaluation DATASET 2
    data = dataholder(
        path = args["data_dir"], 
        b_size = args["batch_size"],
        ood = True,
        ood_dataset_ = 2,
        stage = "train",
        for_rationale = True
    )

    test_predictive_performance(
        test_data_loader = data.test_loader, 
        for_rationale = True, 
        output_dims = data.nu_of_labels,
        save_output_probs = True,
        ood = True,
        ood_dataset_ = 2,
        vocab_size = data.vocab_size
    )   

    del data
    gc.collect()


    ## shows which model performed best on dev F1 (in-domain)
    ## if keep_models = False then will remove the rest of the models to save space
    keep_best_model_(
        keep_models = False,
        for_rationale = True
    )

else:

    if args["train_on_ood"]: 
        ood_train = True
        raise NotImplementedError("""
        Do I need to train on OOD and test performance on OOD when extracting
        rationales using another model?        
        """)
    else: ood_train = False

    train_and_save(
        train_data_loader = data.train_loader, 
        dev_data_loader = data.dev_loader, 
        for_rationale = True, 
        output_dims = data.nu_of_labels,
        ood = ood_train,
        vocab_size = data.vocab_size
    )
