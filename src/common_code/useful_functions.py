# """
# contains functions for helping with the loading, processing, description and preparation of the datasets
# """

import numpy as np
import json 
import torch
import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

def wpiece2word(tokenizer, sentence, weights, print_err = False):  

    """
    converts word-piece ids to words and
    importance scores/weights for word-pieces to importance scores/weights
    for words by aggregating them
    """

    tokens = tokenizer.convert_ids_to_tokens(sentence)

    new_words = {}
    new_score = {}

    position = 0

    for i in range(len(tokens)):

        word = tokens[i]
        score = weights[i]

        if "##" not in word:
            
            position += 1
            new_words[position] = word
            new_score[position] = score
            
        else:
            
            new_words[position] += word.split("##")[1]
            new_score[position] += score

    return np.asarray(list(new_words.values())), np.asarray(list(new_score.values()))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_rationale_mask_(
        importance_scores = torch.tensor, 
        no_of_masked_tokens = np.ndarray,
        method = "topk", batch_input_ids = None
    ):

    rationale_mask = []

    for _i_ in range(importance_scores.size(0)):
        
        score = importance_scores[_i_]
        tokens_to_mask = int(no_of_masked_tokens[_i_])
        
        ## if contigious or not a unigram (unigram == topk of 1)
        if method == "contigious" and tokens_to_mask > 1:

            top_k = contigious_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )
        
        else:

            top_k = topk_(
                importance_scores = score,
                tokens_to_mask = tokens_to_mask
            )

        ## create the instance specific mask
        ## 1 represents the rationale :)
        ## 0 represents tokens that we dont care about :'(
        mask = torch.zeros(score.shape).to(device)
        mask = mask.scatter_(-1,  top_k.to(device), 1).long()

        ## now if we have a query we need to preserve the query in the mask
        if batch_input_ids is not None:
            
            sos_eos = torch.where(batch_input_ids[_i_] == 102)[0]
            seq_length = sos_eos[0]
            query_end = sos_eos[1]

            mask[seq_length: query_end+1] = 1 

        rationale_mask.append(mask)

    rationale_mask = torch.stack(rationale_mask).to(device)

    return rationale_mask

def contigious_(importance_scores, tokens_to_mask):

    ngram = torch.stack([importance_scores[i:i + tokens_to_mask] for i in range(len(importance_scores) - tokens_to_mask + 1)])
    indxs = [torch.arange(i, i+tokens_to_mask) for i in range(len(importance_scores) - tokens_to_mask + 1)]
    top_k = indxs[ngram.sum(-1).argmax()]

    return top_k

def topk_(importance_scores, tokens_to_mask):

    top_k = torch.topk(importance_scores, tokens_to_mask).indices

    return top_k

def batch_from_dict_(batch_data, metadata, target_key = "original prediction", feature_attribution = None):

    new_tensor = []

    for _id_ in batch_data["annotation_id"]:

        new_tensor.append(
            metadata[_id_][target_key]
        )

    return torch.tensor(new_tensor).to(device)

def create_only_query_mask_(
    batch_input_ids : torch.tensor, 
    special_tokens : dict) -> np.array:

    """
    Creates and returns a mask that preserves queries when handling multi-input documents
    Args: 
        batch_input_ids --> shape : [B,S,1]
        special_tokens --> PAD, SEP, CLS token ids
    """

    query_mask = []

    ## loop through each input in the batch
    for seq in batch_input_ids:
        
        ## create an empty mask
        only_query_mask = torch.zeros(seq.shape).to(device)

        ## 
        sos_eos = torch.where(seq == special_tokens["sep_token_id"][0].item())[0]
        seq_length = sos_eos[0] + 1 
        query_end = sos_eos[1]

        only_query_mask[seq_length: query_end+1] = 1 

        query_mask.append(only_query_mask)

    query_mask = torch.stack(query_mask).to(device)

    return query_mask.long()
