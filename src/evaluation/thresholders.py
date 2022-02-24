#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math 
import numpy as np

# class thresholders(object):

def contigious(scores, original_length, rationale_length = 0.1):

    """ 
    Contiguous rationale extractor
    Indicates phrase from input with the highest collective salient scores
    Input: 
        scores : {
            "info" : salient_scores ,
            "size" : sequence length ,
            "options" : normalised attention weights / gradients / or other 
                        token importance metric
            }
        rationale_length : {
            "info" : percentage of original length of sequence to extract
            "size" : between 0 - 1 
            }
    Output:
        rationales : {
            "info" : returns indexes that form the rationale from the sequence
            }
    """

    assert rationale_length > 0. and rationale_length <= 1.
    
    max_length = math.ceil(float(original_length) * rationale_length)

    if max_length == 0: max_length = 1

    nodes = np.stack([scores[i:i + max_length] for i in range(len(scores) - max_length + 1)])
    indxs = [np.arange(i, i+max_length) for i in range(len(scores) - max_length + 1)]

    max_node = np.argmax(nodes.sum(-1))

    return indxs[max_node]

def topk(scores, original_length, rationale_length = 0.1):

    """ 
    topk rationale extractor
    Indicates tokens from input that form the rationale length
    Input: 
        scores : {
            "info" : salient_scores ,
            "size" : sequence length ,
            "options" : normalised attention weights / gradients / or other 
                        token importance metric
            }
        rationale_length : {
            "info" : percentage of original length of sequence to extract
            "size" : between 0 - 1 
            }
    Output:
        rationales : {
            "info" : returns indexes that form the rationale from the sequence
            }
    """

    assert rationale_length > 0. and rationale_length <= 1.

    max_lengths = math.ceil(float(original_length) * rationale_length)

    if max_lengths == 0: max_lengths = 1

    indxs = np.argsort(-scores)[:max_lengths]

    return indxs
