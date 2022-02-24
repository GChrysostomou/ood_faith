import torch
import numpy as np
import json
import config.cfg
from config.cfg import AttrDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))

class predictor:
    
    def __init__(self, model, tokenizer, seq_length):

        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = seq_length

    def convert_text_to_features(self, text):
        
        ## if empty text
        if len(text.split()) == 0: 

            text = "[CLS] [SEP]" 

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(text.split())).unsqueeze(0) #b x s

        if args.query:
            
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            if args.model_abbreviation == "roberta":
                
                token_type_ids = torch.zeros_like(attention_mask)

            else:
                
                if (self.tokenizer.sep_token_id in input_ids.squeeze(0)) and (input_ids.size(1) > 2):

                    sos, eos = torch.where(input_ids == self.tokenizer.sep_token_id)[1]

                    token_type_ids = torch.zeros_like(attention_mask)
                    token_type_ids[0, sos:eos+1] = 1

                else:

                    token_type_ids = torch.zeros_like(attention_mask)

        else:

            token_type_ids = (input_ids != self.tokenizer.pad_token_id).long()
            attention_mask = token_type_ids.clone()

        
        return {"input_ids" : input_ids, "token_type_ids" : token_type_ids, "attention_mask" : attention_mask}

    def predictor(self, text):

        examples = []

        for example in text:
            examples.append(self.convert_text_to_features(example))

        results = []

        for example in examples:

            batch = {
                "input_ids" : example["input_ids"].to(device),
                "token_type_ids" : example["token_type_ids"].to(device),
                "attention_mask" : example["attention_mask"].to(device),
                "retain_gradient" : False
            }

            with torch.no_grad():
                
                outputs = self.model(**batch)
            
            logits = outputs[0]
            logits = torch.softmax(logits, dim = 1)
            results.append(logits.cpu().detach().numpy()[0])

        results_array = np.array(results)

        return results_array