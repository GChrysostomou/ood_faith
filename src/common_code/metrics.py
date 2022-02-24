import numpy as np
import json
import torch
import config.cfg
from config.cfg import AttrDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class uncertainty_metrics:

    def __init__(self, data, save_dir = None, ood = False, ood_dataset_ = 0):

        y_prob, y_true = zip(*[(x["predicted"], float(x["actual"])) for x in data.values()])
        # self.y_prob = np.asarray(y_prob)
        self.y_prob = np.asarray([softmax(x) for x in y_prob])
        self.y_true = np.asarray(y_true)
        self.save_dir = save_dir
        self.ood = ood
        self.ood_dataset_ = ood_dataset_

    def ece(self,n_bins=10):

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        confidences, predictions = np.max(self.y_prob, axis=1), np.argmax(self.y_prob, axis=1)
        accuracies = np.equal(predictions, self.y_true)
        ece, refinement = 0.0, 0.0

        bin = 0
        bin_stats = {}
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin += 1
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences, bin_lower) * np.less_equal(confidences, bin_upper)
            prop_in_bin = np.mean(in_bin*1.0)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.absolute(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                refinement += (accuracy_in_bin * (1 - accuracy_in_bin)) * prop_in_bin
                # refinement += (accuracy_in_bin * (1-accuracy_in_bin)) * prop_in_bin
            else:
                avg_confidence_in_bin = 0.
                accuracy_in_bin = 0.
            bin_stats[bin] = {'acc': float(round(accuracy_in_bin,3)), 'conf': float(round(avg_confidence_in_bin,3))}

        total = ece + refinement
        

        fname = self.save_dir + "_ece-stats.json"
        
        if self.ood: 
            
            ood_name = args.ood_dataset_1 if self.ood_dataset_ == 1 else args.ood_dataset_2

            fname = self.save_dir + f"_OOD_{ood_name}_ece-stats.json"

        with open(fname, 'w') as file:

            json.dump(
                {'ece': ece, "refinement": refinement, "bins": bin_stats},
                file,
                indent = 4
            ) 

        return {'ece': ece, "refinement": refinement, "bins": bin_stats}

    def entropy_(self,return_vec=False):
        ent = -1.0 * np.sum(np.multiply(self.y_prob, np.log(self.y_prob + np.finfo(float).eps)), axis=1) / np.log(2)
        ent_mean = np.mean(ent)
        ent_var = np.var(ent)
        if return_vec:
            return ent

        fname = self.save_dir + "_ece-stats.json"
        
        if self.ood: 

            fname = self.save_dir + "_ood_ece-stats.json"

        with open(fname, 'w') as file:
            json.dump(
                {'mean': float(round(ent_mean,3)), 'var': float(round(ent_var,3))},
                file,
                indent = 4
            ) 

        return {'mean': float(round(ent_mean,3)), 'var': float(round(ent_var,3))}



"""
Faithfulness metrics
"""

def sufficiency_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    sufficiency = 1 - np.maximum(0, full_text_probs - reduced_probs)

    return sufficiency

def normalized_sufficiency_(model, original_sentences : torch.tensor, rationale_mask : torch.tensor, 
                            inputs : dict, full_text_probs : np.array, full_text_class : np.array, rows : np.array, 
                            suff_y_zero : np.array) -> np.array:

    ## for sufficiency we always keep the rationale
    ## since ones represent rationale tokens
    ## preserve cls
    rationale_mask[:,0] = 1
    ## preserve sep
    rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1
    
    inputs["input_ids"]  =  rationale_mask[:,:original_sentences.size(1)] * original_sentences

    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat.detach().cpu(), dim = -1).numpy()

    reduced_probs = yhat[rows, full_text_class]

    ## reduced input sufficiency
    suff_y_a = sufficiency_(full_text_probs, reduced_probs)

    # return suff_y_a
    suff_y_zero -= 1e-4 ## to avoid nan

    norm_suff = np.maximum(0, (suff_y_a - suff_y_zero) / (1 - suff_y_zero))

    norm_suff = np.clip( norm_suff, a_min = 0, a_max = 1)

    return norm_suff, reduced_probs

def comprehensiveness_(full_text_probs : np.array, reduced_probs : np.array) -> np.array:

    comprehensiveness = np.maximum(0, full_text_probs - reduced_probs)

    return comprehensiveness

def normalized_comprehensiveness_(model, original_sentences : torch.tensor, rationale_mask : torch.tensor, 
                                  inputs : dict, full_text_probs : np.array, full_text_class : np.array, rows : np.array, 
                                  suff_y_zero : np.array) -> np.array:
    
    ## for comprehensivness we always remove the rationale and keep the rest of the input
    ## since ones represent rationale tokens, invert them and multiply the original input
    rationale_mask = (rationale_mask == 0)
    ## preserve cls
    rationale_mask[:,0] = 1
    ## preserve sep
    rationale_mask[torch.arange(rationale_mask.size(0)).to(device), inputs["lengths"]] = 1

    inputs["input_ids"] =  original_sentences * rationale_mask.long()[:,:original_sentences.size(1)]
    
    yhat, _  = model(**inputs)

    yhat = torch.softmax(yhat, dim = -1).detach().cpu().numpy()

    reduced_probs = yhat[rows, full_text_class]

     ## reduced input sufficiency
    comp_y_a = comprehensiveness_(full_text_probs, reduced_probs)

    # return comp_y_a
    suff_y_zero -= 1e-4 # to avoid nan

    ## 1 - suff_y_0 == comp_y_1
    norm_comp = np.maximum(0, comp_y_a / (1 - suff_y_zero))

    norm_comp = np.clip(norm_comp, a_min = 0, a_max = 1)

    return norm_comp, yhat
    