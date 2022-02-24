import spacy
import re
import torch

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

def cleaner(text, spacy=True) :
    text = re.sub(r'\s+', ' ', text.strip())
    if spacy :
        text = [t.text.lower() for t in nlp(text)]
    else :
        text = [t.lower() for t in text.split()]
    text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
    
    return text

def tokenize(text) :
    text = " ".join(text)
    text = text.replace("-LRB-", '')
    text = text.replace("-RRB-", " ")
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    tokens = " ".join([t.text.lower() for t in nlp(text)])
    return tokens

def invert_and_join(X, idx_to_word) :
        X = [[idx_to_word[x] for x in doc] for doc in X]
        X = [" ".join(x) for x in X]
        return X

def encode_plusplus_(data_dict, tokenizer, max_length, *arguments):

    """
    returns token type ids, padded doc and 
    """

    ## if len(args)  > 1 that means that we have ctx + query
    if len(arguments) > 1:

        model_inputs = tokenizer.encode_plus(
            arguments[0], 
            arguments[1],
            add_special_tokens = True,
            max_length = max_length,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True,
            return_tensors = "pt"             
        )

        data_dict.update(model_inputs)

        del data_dict["document"]
        del data_dict["query"]

        ## query mask used_only for rationale extraction and for masking importance metrics
        ## i.e. keeping only the contxt not the query
        init_mask_ = torch.where(model_inputs["input_ids"] == tokenizer.sep_token_id)[1][0]
        fin_mask = model_inputs["input_ids"].size(-1)
        range_to_zero = torch.arange(init_mask_, fin_mask)
        model_inputs["query_mask"] = model_inputs["attention_mask"].clone()
        model_inputs["query_mask"].squeeze(0)[range_to_zero] = 0
        ## preserve cls token
        model_inputs["query_mask"].squeeze(0)[0] = 0
        

    else:
  
        model_inputs = tokenizer.encode_plus(
            arguments[0], 
            add_special_tokens = True,
            max_length = max_length,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True,
            return_tensors = "pt"             
        )

        del data_dict["text"]
    
        init_mask_ = torch.where(model_inputs["input_ids"] == tokenizer.sep_token_id)[1][0]
        model_inputs["query_mask"] = model_inputs["attention_mask"].clone()
        ## preserve cls token
        model_inputs["query_mask"].squeeze(0)[0] = 0

    ## context length
    model_inputs["lengths"] = init_mask_
    model_inputs["special tokens"] = {
        "pad_token_id" : tokenizer.pad_token_id,
        "sep_token_id" : tokenizer.sep_token_id
    }

    data_dict.update(model_inputs)

    return data_dict
