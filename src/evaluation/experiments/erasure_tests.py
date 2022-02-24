import torch
import torch.nn as nn
import numpy as np 
import json
from tqdm import trange
import numpy as np


import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nn.deterministic = True
torch.backends.cudnn.benchmark = False
    
from src.common_code.useful_functions import create_rationale_mask_, create_only_query_mask_, batch_from_dict_
from src.common_code.metrics import normalized_comprehensiveness_, normalized_sufficiency_, sufficiency_

torch.manual_seed(25)
torch.cuda.manual_seed(25)
np.random.seed(25)

import os

def conduct_experiments_(model, data, model_random_seed, ood = False, ood_dataset_ = None):

    ## now to create folder where results will be saved
    fname = os.path.join(
        os.getcwd(),
        args["extracted_rationale_dir"],
        "importance_scores",
        ""
    )

    os.makedirs(fname, exist_ok = True)

    if ood: fname = f"{fname}test_importance_scores-OOD-{ood_dataset_}-{model_random_seed}.npy"
    else: fname = f"{fname}test_importance_scores-{model_random_seed}.npy"

    ## retrieve importance scores
    importance_scores = np.load(fname, allow_pickle = True).item()

    if ood:  desc = f'faithfulness evluation -> OOD-{ood_dataset_}'
    else: desc = 'faithfulness evaluation -> id'
    
    pbar = trange(len(data) * data.batch_size, desc=desc, leave=True)
    
    faithfulness_results = {}

    if ood_dataset_:

        desired_rationale_length = args.ood_rat_1 if ood_dataset_ == args.ood_dataset_1 else args.ood_rat_2

    else:

        desired_rationale_length = args.rationale_length

    print(f"*** desired_rationale_length --> {desired_rationale_length}")

    for batch in data:
        
        model.eval()
        model.zero_grad()

        batch = {
                "annotation_id" : batch["annotation_id"],
                "input_ids" : batch["input_ids"].squeeze(1).to(device),
                "lengths" : batch["lengths"].to(device),
                "labels" : batch["label"].to(device),
                "token_type_ids" : batch["token_type_ids"].squeeze(1).to(device),
                "attention_mask" : batch["attention_mask"].squeeze(1).to(device),
                "query_mask" : batch["query_mask"].squeeze(1).to(device),
                "special_tokens" : batch["special tokens"],
                "retain_gradient" : False
            }
            
        assert batch["input_ids"].size(0) == len(batch["labels"]), "Error: batch size for item 1 not in correct position"
   
        original_prediction, _ =  model(**batch)

        original_prediction.max(-1)[0].sum().backward(retain_graph = True)

        ## setting up the placeholder for storing the results
        for annot_id in batch["annotation_id"]:
            faithfulness_results[annot_id] = {}

        ## prepping for our experiments
        rows = np.arange(batch["input_ids"].size(0))

        original_sentences = batch["input_ids"].clone().detach()

        original_prediction = torch.softmax(original_prediction, dim = -1).detach().cpu().numpy().astype(np.float64)

        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)

        original_sentences = batch["input_ids"].clone()

        rows = np.arange(original_sentences.size(0))
        
        ## now measuring baseline sufficiency for all 0 rationale mask
        if args.query:

            only_query_mask=create_only_query_mask_(
                batch_input_ids=batch["input_ids"],
                special_tokens=batch["special_tokens"]
            )

            batch["input_ids"] = only_query_mask * original_sentences

        else:

            only_query_mask=torch.zeros_like(batch["input_ids"]).long()

            batch["input_ids"] = only_query_mask


        yhat, _  = model(**batch)

        yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

        reduced_probs = yhat[rows, full_text_class]

        ## baseline sufficiency
        suff_y_zero = sufficiency_(
            full_text_probs, 
            reduced_probs
        )

        ## AOPC scores and other metrics
        rationale_ratios = [0.02, 0.1, 0.2, 0.5]

        for rationale_type in {args.thresholder}:

            for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                faithfulness_results[annot_id]["full text prediction"] = original_prediction[_j_] 
                faithfulness_results[annot_id]["true label"] = batch["labels"][_j_].detach().cpu().item()
            
            for feat_name in {"random", "attention", "gradients", "ig" , "scaled attention", "lime", "deeplift"}:

                feat_score =  batch_from_dict_(
                    batch_data = batch, 
                    metadata = importance_scores, 
                    target_key = feat_name,
                )

                suff_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)
                comp_aopc = np.zeros([yhat.shape[0], len(rationale_ratios)], dtype=np.float64)

                for _i_, rationale_length in enumerate(rationale_ratios):
                    
                    ## if we are masking for a query that means we are preserving
                    ## the query and we DO NOT mask it
                    if args.query:

                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = rationale_type,
                            batch_input_ids = original_sentences
                        )

                    else:

                        rationale_mask = create_rationale_mask_(
                            importance_scores = feat_score, 
                            no_of_masked_tokens = torch.ceil(batch["lengths"].float() * rationale_length).detach().cpu().numpy(),
                            method = rationale_type
                        )

                    ## measuring faithfulness
                    comp, comp_probs  = normalized_comprehensiveness_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero
                    )

                    suff, suff_probs = normalized_sufficiency_(
                        model = model, 
                        original_sentences = original_sentences, 
                        rationale_mask = rationale_mask, 
                        inputs = batch, 
                        full_text_probs = full_text_probs, 
                        full_text_class = full_text_class, 
                        rows = rows,
                        suff_y_zero = suff_y_zero
                    )

                    suff_aopc[:,_i_] = suff
                    comp_aopc[:,_i_] = comp
                    
                    ## store the ones for the desired rationale length
                    ## the rest are just for aopc
                    if rationale_length == desired_rationale_length:

                        sufficiency = suff
                        comprehensiveness = comp
                        comp_probs_save = comp_probs
                        suff_probs_save = suff_probs
                    
                for _j_, annot_id in enumerate(batch["annotation_id"]):
                    
                    faithfulness_results[annot_id][feat_name] = {
                        f"sufficiency @ {desired_rationale_length}" : sufficiency[_j_],
                        f"comprehensiveness @ {desired_rationale_length}" : comprehensiveness[_j_],
                        f"masked R probs (comp) @ {desired_rationale_length}" : comp_probs_save[_j_].astype(np.float64),
                        f"only R probs (suff) @ {desired_rationale_length}" : suff_probs_save[_j_].astype(np.float64),
                        "sufficiency aopc" : {
                            "mean" : suff_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : suff_aopc[_j_]
                        },
                        "comprehensiveness aopc" : {
                            "mean" : comp_aopc[_j_].sum() / (len(rationale_ratios) + 1),
                            "per ratio" : comp_aopc[_j_]
                        }
                    }
        

            
           
        pbar.update(data.batch_size)

            
    descriptor = {}
    # filling getting averages
    for feat_attr in {"attention", "gradients", "ig", "random", "scaled attention", "lime", "deeplift"}:
        
        sufficiencies = np.asarray([faithfulness_results[k][feat_attr][f"sufficiency @ {desired_rationale_length}"] for k in faithfulness_results.keys()])
        comprehensivenesses = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness @ {desired_rationale_length}"] for k in faithfulness_results.keys()])
        aopc_suff= np.asarray([faithfulness_results[k][feat_attr][f"sufficiency aopc"]["mean"] for k in faithfulness_results.keys()])
        aopc_comp = np.asarray([faithfulness_results[k][feat_attr][f"comprehensiveness aopc"]["mean"] for k in faithfulness_results.keys()])

        descriptor[feat_attr] = {
            "sufficiency" : {
                "mean" : sufficiencies.mean(),
                "std" : sufficiencies.std()
            },
            "comprehensiveness" : {
                "mean" : comprehensivenesses.mean(),
                "std" : comprehensivenesses.std()
            },
            "AOPC - sufficiency" : {
                "mean" : aopc_suff.mean(),
                "std" : aopc_suff.std()
            },
            "AOPC - comprehensiveness" : {
                "mean" : aopc_comp.mean(),
                "std" : aopc_comp.std()
            }
        }

    ## save all info
    fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-detailed-" + str(model_random_seed) + ".npy"

    if ood: fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-detailed-OOD-{ood_dataset_}-" + str(model_random_seed) + ".npy"

    np.save(fname, faithfulness_results)

    ## save descriptors
    fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-averages-" + str(model_random_seed) + "-description.json"

    if ood: fname = args["evaluation_dir"] + f"{args.thresholder}-faithfulness-scores-averages-OOD-{ood_dataset_}-" + str(model_random_seed) + "-description.json"


    with open(fname, 'w') as file:
            json.dump(
                descriptor,
                file,
                indent = 4
            ) 

    return
