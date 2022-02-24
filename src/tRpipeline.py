"""
This module contains functions that:
train_and_save : function that will train on user defined runs
                 the predefined model on a user selected dataset. 
                 Will save details of model training and development performance
                 for each run and epoch. Will save the best model for each run
test_predictive_performance : function that obtains a trained model
                              on a user defined task and model. Will 
                              test on the test-dataset and will keep 
                              the best performing model, whilst also returning
                              statistics for model performances across runs, mean
                              and standard deviations
"""


import torch
import torch.nn as nn
import json 
import numpy as np
import os
from transformers.optimization import AdamW
from torch.optim import Adam
import logging
import gc
import config.cfg
from config.cfg import AttrDict
from config.inherent_config import kuma_args, rl_args, full_lstm_args
with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from src.models.deterministic.bert import BertClassifier
from src.models.deterministic.tasc import lin as tasc
from src.models.stochastic.kuma_model import LatentRationaleModel
from src.models.deterministic.lstm import LSTMClassifier
from src.models.stochastic.rl import RLModel
from src.common_code.train_test import train_model, test_model


## select model depending on if normal bert
## or rationalizer 

faithful_variant = {
    "kuma" : LatentRationaleModel,
    "rl" : RLModel,
    "full_lstm" : LSTMClassifier
}

if kuma_args.__name__ == args.inherently_faithful: 
    faith_args = kuma_args
elif rl_args.__name__ == args.inherently_faithful: 
    faith_args = rl_args
else: faith_args = full_lstm_args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_save(train_data_loader, dev_data_loader, for_rationale = False, output_dims = 2, ood = False, vocab_size = None):

  
    """
    Trains the models depending on the number of random seeds
    a user supplied, saves the best performing models depending
    on dev loss and returns also stats
    """


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    np.random.seed(args["seed"])


    if args.use_tasc and args.stage_of_proj == "train":
            
        tasc_variant = tasc
        
        tasc_mech = tasc_variant(vocab_size)
        
        print("*** TASC MODEL ****")

    else:
        
        tasc_mech = None
    
        print("*** VANILLA MODEL ****")
    
    if args.inherently_faithful is None:

        classifier = BertClassifier(
            output_dim = output_dims,
            tasc = tasc_mech
        )

    else:

        classifier = faithful_variant[args.inherently_faithful](
            output_dim = output_dims,
            tasc = tasc_mech,
            **faith_args.get_[args.dataset]["MODEL_ARGS_"]
        )

    classifier.to(device)

    loss_function = nn.CrossEntropyLoss() 

    if args.inherently_faithful: 

        optimiser = Adam(
            params = classifier.parameters(),
            **faith_args.get_[args.dataset]["OPTIM_ARGS_"]
        )


    else:

        if args.use_tasc:
            
            classifier.wrapper.model.embeddings.word_embeddings.weight.requires_grad_(False)

            difference = sum(p.numel() for p in classifier.parameters() if p.requires_grad) + classifier.wrapper.model.embeddings.word_embeddings.weight.numel()

            assert difference == sum(p.numel() for p in classifier.parameters()), ("""
            embeddings not frozen properly and will be used in the optimizer
            """)

            optimiser = AdamW([
                {'params': [p for n,p in classifier.wrapper.named_parameters() if p.requires_grad], 'lr': args.lr_bert},
                {'params': classifier.output_layer.parameters(), 'lr': args.lr_classifier},
                {'params': [p for n,p in classifier.named_parameters() if "u_param" in n], 'lr': args.lr_classifier}],  
                correct_bias = False
            )

        else:

            assert sum(p.numel() for p in classifier.parameters() if p.requires_grad) == sum(p.numel() for p in classifier.parameters()), ("""
            some of the parameters in this model are not trainable (Note: They should all be trainable)
            """)

            optimiser = AdamW([
                {'params': [p for p in classifier.wrapper.parameters() if p.requires_grad], 'lr': args.lr_bert},
                {'params': classifier.output_layer.parameters(), 'lr': args.lr_classifier}], 
                correct_bias = False
            )

    if for_rationale:

        saving_model = os.path.join(
            args["model_dir"], 
            args["thresholder"],
            f"{args.importance_metric}_{args.model_abbreviation}{args.seed}.pt"
        )

    else:

        saving_model = os.path.join(
            args["model_dir"],
            f"{args.model_abbreviation}{args.seed}.pt"
        )

    _, results_to_save = train_model(
        classifier,  
        train_data_loader, 
        dev_data_loader, 
        loss_function,
        optimiser,
        epochs = args["epochs"],
        cutoff = False, 
        save_folder = saving_model,
        run = str(args["seed"]),
        seed = str(args["seed"])
    )

    if for_rationale:


        text_file = open(
            os.path.join(
                args["model_dir"], 
                args["thresholder"],
                f"model_run_stats/{args.importance_metric}_{args.model_abbreviation}_seed_{args.seed}.txt"
            ), 
        "w")

    else:

        text_file = open(
            os.path.join(
                args["model_dir"], 
                f"model_run_stats/{args.model_abbreviation}_seed_{args.seed}.txt"
            ), 
        "w")

    text_file.write(results_to_save)
    text_file.close()

    del classifier
    gc.collect()
    torch.cuda.empty_cache()

    return

import glob
import os 
import re
from src.common_code.metrics import uncertainty_metrics


def test_predictive_performance(
    test_data_loader, for_rationale = False, output_dims = 2, 
    save_output_probs = True, ood = False, ood_dataset_ = 0, vocab_size = None):    

    """
    Runs trained models on test set
    Also keeps the best model for experimentation
    and produces statistics    
    """
    
    if for_rationale: trained_models = glob.glob(os.path.join(args["model_dir"], args["thresholder"],"") + args["importance_metric"] + "*.pt")
    else: trained_models = glob.glob(args["model_dir"] + args["model_abbreviation"] +"*.pt")
    
    stats_report = {}

    logging.info("-------------------------------------")
    logging.info("evaluating trained models")
    
    for model in trained_models:
        
        if args.use_tasc and args.stage_of_proj == "train":
            
            tasc_variant = tasc
            
            tasc_mech = tasc_variant(vocab_size)
            
        else:
            
            tasc_mech = None
        
        if args.inherently_faithful is None:

            classifier = BertClassifier(
                output_dim = output_dims,
                tasc = tasc_mech
            )

        else:

            classifier = faithful_variant[args.inherently_faithful](
                output_dim = output_dims,
                tasc = tasc_mech,
                **faith_args.get_[args.dataset]["MODEL_ARGS_"]
            )

        classifier.to(device)
    
        classifier.load_state_dict(torch.load(model, map_location=device))
        
        logging.info(
            "Loading model: {}".format(
                model
            )
        )

        classifier.to(device)
        
        seed = re.sub("bert", "", model.split(".pt")[0].split("/")[-1])

        if args.inherently_faithful: seed = seed = re.sub("faith-bert", "", model.split(".pt")[0].split("/")[-1])

        loss_function = nn.CrossEntropyLoss()

        test_results,test_loss, test_predictions = test_model(
                model =classifier, 
                loss_function = loss_function, 
                data= test_data_loader,
                save_output_probs = save_output_probs,
                random_seed = seed,
                for_rationale = for_rationale,
                ood = ood,
                ood_dataset_ = ood_dataset_
            )

        ## save stats of evaluated model
        if for_rationale:

            if ood:

                assert ood_dataset_ in [1,2], (
                    f"""
                    Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                    """
                )

                ood_name = args.ood_dataset_1 if ood_dataset_ == 1 else args.ood_dataset_2
                
                fname = os.path.join(
                    args["model_dir"],
                    args["thresholder"],
                    "model_run_stats",
                    f"{args.importance_metric}_{args.model_abbreviation}_best_model_test_seed:" + seed + f"-OOD-{ood_name}.json"
                )

            else:

                fname = os.path.join(
                    args["model_dir"],
                    args["thresholder"],
                    "model_run_stats",
                    f"{args.importance_metric}_{args.model_abbreviation}_best_model_test_seed:" + seed + ".json"
                )
            
            
        else:

            fname = os.path.join(
                args["model_dir"],
                "model_run_stats",
                f"test-stats-{args.model_abbreviation}:{seed}.json"
            )

            if ood: 
                
                assert ood_dataset_ in [1,2], (
                    f"""
                    Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                    """
                )

                ood_name = args.ood_dataset_1 if ood_dataset_ == 1 else args.ood_dataset_2

                fname = os.path.join(
                    args["model_dir"],
                    "model_run_stats",
                    f"test-stats-{args.model_abbreviation}:{seed}-OOD-{ood_name}.json"
                )

        logging.info(
            "Seed: '{0}' -- Test loss: '{1}' -- Test accuracy: '{2}'".format(
                seed, 
                round(test_loss, 3),
                round(test_results["macro avg"]["f1-score"], 3)
            )
        )

        with open(fname, "w") as file: json.dump(
            test_results,
            file,
            indent = 4
        )
        del classifier
        gc.collect()
        torch.cuda.empty_cache()

        ### conducting ece test
        unc_metr = uncertainty_metrics(
            data = test_predictions, 
            save_dir = model.split(".")[0], #remove .pt
            ood = ood,
            ood_dataset_ = ood_dataset_
        )

        ece_stats = unc_metr.ece()

        stats_report[seed] = {}
        stats_report[seed]["model"] = model
        stats_report[seed]["f1"] = test_results["macro avg"]["f1-score"]
        stats_report[seed]["accuracy"] = test_results["accuracy"]
        stats_report[seed]["loss"] = test_loss
        stats_report[seed]["ece-score"] = ece_stats["ece"]

    f1s = np.asarray([x["f1"] for k,x in stats_report.items()])
    eces = np.asarray([x["ece-score"] for k,x in stats_report.items()])
    accs = np.asarray([x["accuracy"] for k,x in stats_report.items()])

    stats_report["mean-acc"] = accs.mean()
    stats_report["std-acc"] = accs.std()

    stats_report["mean-f1"] = f1s.mean()
    stats_report["std-f1"] = f1s.std()
    
    stats_report["mean-ece"] = eces.mean()
    stats_report["std-ece"] = eces.std()

    if for_rationale:

        if ood: 
            
            assert ood_dataset_ in [1,2], (
                f"""
                Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                """
            )

            ood_name = args.ood_dataset_1 if ood_dataset_ == 1 else args.ood_dataset_2

            fname = os.path.join(
                args["model_dir"], 
                args["thresholder"],
                f"{args.importance_metric}_{args.model_abbreviation}_predictive_performances-OOD-{ood_name}.json"
            )

        else:
            
            fname = os.path.join(
                args["model_dir"], 
                args["thresholder"],
                f"{args.importance_metric}_{args.model_abbreviation}_predictive_performances.json"
            )

    else:

        fname =  os.path.join(
            args["model_dir"],
            f"{args.model_abbreviation}_predictive_performances.json"
        )

        if ood:

            assert ood_dataset_ in [1,2], (
                f"""
                Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                """
            )

            ood_name = args.ood_dataset_1 if ood_dataset_ == 1 else args.ood_dataset_2

            fname =  os.path.join(
                args["model_dir"],
                f"{args.model_abbreviation}_predictive_performances-OOD-{ood_name}.json"
            )

    with open(fname, 'w') as file:
        json.dump(
            stats_report,
            file,
            indent = 4
        )

    return stats_report

def keep_best_model_(keep_models = False, for_rationale = False):

    if for_rationale:

        dev_stats = glob.glob(
            os.path.join(
                args["model_dir"], 
                args["thresholder"],
                "model_run_stats/*dev*.json"
            )
        )

        dev_stats = [x for x in dev_stats if f"/{args.importance_metric}" in x]

    else:

        dev_stats = glob.glob(
            os.path.join(
                args["model_dir"], 
                "model_run_stats/*dev*.json"
            )
        )

    if args.use_tasc: dev_stats = [x for x in dev_stats if "tasc" in x]
    else: dev_stats = [x for x in dev_stats if "tasc" not in x]

    dev_stats_cleared = {}

    for stat in dev_stats:
        
        with open(stat) as file: stats = json.load(file)
        dev_loss = stats["dev_loss"]
        ## use f1 of devset for keeping models
        dev_stats_cleared[stats['model_name']] = dev_loss

    best_model, _ = zip(*sorted(dev_stats_cleared.items(), key=lambda item: item[1], reverse = True))

    print("*** best model on dev F1 is {}".format(best_model[-1]))  

    if keep_models == False:

        ## if its the rationale models we are not interested in saving them
        if for_rationale:

            to_rm_models = dev_stats_cleared.keys()

        else:

            to_rm_models, _ = zip(*sorted(dev_stats_cleared.items(), key=lambda item: item[1], reverse = True)[:-1])

        for rmvd_model in to_rm_models:
            
            print("*** removing model {}".format(rmvd_model))

            os.remove(rmvd_model)

    return

    