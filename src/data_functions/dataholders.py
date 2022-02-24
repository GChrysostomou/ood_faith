from src.data_functions.cleaners_encoders import encode_plusplus_
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import re
import pandas as pd
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import config.cfg
from config.cfg import AttrDict

with open(config.cfg.config_directory + 'instance_config.json', 'r') as f:
    args = AttrDict(json.load(f))


## holds our data
class BERT_HOLDER():
    """
    class that holds our data, pretrained tokenizer and set sequence length 
    for a classification task
    """
    def __init__(self, path = str, b_size = 8 , 
                for_rationale = False, ood = False, ood_dataset_ = 0,
                stage = "train", return_as_frames = False):
  
        """
        loads data for a classification task from preprocessed .csv 
        files in the dataset/data folder
        and returns three dataholders : train, dev, test
        """

        assert type(b_size) == int
    
        self.batch_size = b_size

        ## if loading rationales we have to also include the importance metric
        if for_rationale:
            
            if ood: 

                assert ood_dataset_ in [1,2], (
                    f"""
                    Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                    """
                )

                ood_name = args.ood_dataset_1 if ood_dataset_ == 1 else args.ood_dataset_2
        
                path += args["thresholder"] +  f"/OOD-{ood_name}-" + args["importance_metric"] +  "-"

                print(f"**** Loading dataset OOD --> {path}")
                logging.info(f"**** Loading dataset OOD --> {path}")
            
            else: 
            
                if args.use_tasc: args["importance_metric"] = "tasc_" + args["importance_metric"]
                path += args["thresholder"] + "/" + args["importance_metric"] + "-"

        else:

            if ood:
                
                assert ood_dataset_ in [1,2], (
                    f"""
                    Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                    """
                )

                if ood_dataset_ == 1:
                    
                    path = re.sub(args["dataset"], args["ood_dataset_1"], path)
                    print(f"**** Loading dataset OOD --> {args.ood_dataset_1}")
                    logging.info(f"**** Loading dataset OOD --> {args.ood_dataset_1}")

                elif ood_dataset_ == 2:

                    path = re.sub(args["dataset"], args["ood_dataset_2"], path)
                    print(f"**** Loading dataset OOD --> {args.ood_dataset_2}")
                    logging.info(f"**** Loading dataset OOD --> {args.ood_dataset_2}")

        ## load data
        with open(f"{path}train.json", "r") as file: train = json.load(file)#[:32]
        with open(f"{path}dev.json", "r") as file: dev = json.load(file)#[:32]
        with open(f"{path}test.json", "r") as file: test = json.load(file)#[:32]

        print("*** loading data in dataholder")

        ## if we are dealing with a query we need to account for the query length as well
        if args.query:
            
            max_len = round(max([len(x["document"].split()) for x in train])) + \
                        max([len(x["query"].split()) for x in train])
            max_len = round(max_len)

        else:
            
            max_len = round(max([len(x["text"].split()) for x in train]))

        self.max_len = min(max_len, 256)

        # load the pretrained tokenizer
        pretrained_weights = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

        self.nu_of_labels = len(np.unique([x["label"] for x in train]))
        self.vocab_size = len(self.tokenizer.vocab)

        if args.query:
            
            train = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["document"], dic["query"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["document"], dic["query"]) for dic in dev]
            test = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["document"], dic["query"]) for dic in test]

        else:

            train = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in train]
            dev = [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in dev]
            test= [encode_plusplus_(dic, self.tokenizer, self.max_len,  dic["text"]) for dic in test]

        shuffle_during_iter = True

        if stage != "train": 

            # sort by length for evaluation and rationale extraction
            train = sorted(train, key = lambda x : x["lengths"], reverse = False)
            dev = sorted(dev, key = lambda x : x["lengths"], reverse = False)
            test = sorted(test, key = lambda x : x["lengths"], reverse = False)

            shuffle_during_iter = False
        
        if return_as_frames:

            self.return_as_frames = {
                "train" : pd.DataFrame(train),
                "dev" : pd.DataFrame(dev),
                "test" : pd.DataFrame(test)
            }

        # prepare data-loaders for training
        self.train_loader = DataLoader(
            train,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False
        )

        self.dev_loader = DataLoader(
            dev,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False
        )

        self.test_loader = DataLoader(
            test,
            batch_size = self.batch_size,
            shuffle = shuffle_during_iter,
            pin_memory = False
        )  

        print("*** dataholder ready")

    def as_dataframes_(self):

        return self.return_as_frames 

from src.data_functions.non_transformer import pretrained_embeds, extract_vocabulary_
import os
## holds our data
class KUMA_RL_HOLDER():
    
    """
    Data holder for our inherenlty faithful models
    RL + KUMA    
    """

    def __init__(self, path : str, b_size : int =  8 , 
                ood : bool = False, ood_dataset_ : int = 0):
  
        assert type(b_size) == int
    
        self.batch_size = b_size

        
        if ood:
            
            assert ood_dataset_ in [1,2], (
                f"""
                Must specify either to use OOD dataset 1 or 2 not {ood_dataset_}    
                """
            )

            if ood_dataset_ == 1:
                
                path = re.sub(args["dataset"], args["ood_dataset_1"], path)
                print(f"**** Loading dataset OOD --> {args.ood_dataset_1}")
                logging.info(f"**** Loading dataset OOD --> {args.ood_dataset_1}")

            elif ood_dataset_ == 2:

                path = re.sub(args["dataset"], args["ood_dataset_2"], path)
                print(f"**** Loading dataset OOD --> {args.ood_dataset_2}")
                logging.info(f"**** Loading dataset OOD --> {args.ood_dataset_2}")

        ## load data
        with open(f"{path}train.json", "r") as file: train = json.load(file)#[:32]
        with open(f"{path}dev.json", "r") as file: dev = json.load(file)#[:32]
        with open(f"{path}test.json", "r") as file: test = json.load(file)#[:32]

        print("*** loading data in dataholder")
            
        self.max_len = round(max([len(x["text"].split()) for x in train]))
        self.nu_of_labels = len(np.unique([x["label"] for x in train]))

        ## check if we processed already the pretrained embeds and vocab
        vocab_fname = os.path.join(
            args.data_dir,
            "vocabulary.json"
        )

        if os.path.exists(vocab_fname):

            with open(vocab_fname, "r") as f: self.w2ix = json.load(f)

        else:

            self.w2ix = extract_vocabulary_(
                data = train
            )

            with open(vocab_fname, "w") as f: 
                json.dump(
                    {k:int(v) for k,v in self.w2ix.items()},
                    f
                )

        self.vocab_size = len(self.w2ix)

        embed_fname = os.path.join(
            args.data_dir,
            f"{args.embed_model}_embeds.npy"
        )
        if os.path.exists(embed_fname):

            pass

        else: ## if not lets create them and save them

            ix2w = {v:k for k,v in self.w2ix.items()}

            embeds = pretrained_embeds(
                model = args.embed_model,
                ix_to_word=ix2w
            ).processed()

            np.save(
                embed_fname,
                embeds
            )

        train = self._process_data_(train)
        dev = self._process_data_(dev)
        test = self._process_data_(test)

        # prepare data-loaders for training
        self.train_loader = DataLoader(
            train,
            batch_size = self.batch_size,
            shuffle = True,
            pin_memory = False
        )

        self.dev_loader = DataLoader(
            dev,
            batch_size = self.batch_size,
            pin_memory = False
        )

        self.test_loader = DataLoader(
            test,
            batch_size = self.batch_size,
            pin_memory = False
        )  

    def _process_data_(self, data_to_process):
        
        return [self._process_instance_(x) for x in data_to_process]
        

    def _process_instance_(self, instance):
        
        instance["input_ids"] = [self.w2ix["<SOS>"]] + \
            [self.w2ix[w] if w in self.w2ix else self.w2ix["<UNKN>"] for w in instance["text"].split(" ")] + \
                [self.w2ix["<EOS>"]]
        
        # del instance["text"]
        instance["length"] = len(instance["input_ids"])
        instance["input_ids"] = np.asarray(
            self._pad_data_(
                tokenized_ids = instance["input_ids"], 
                pad_length = self.max_len
            )
        )

        if instance["input_ids"][-1] != 0:
            
            instance["input_ids"][-1] = self.w2ix["<EOS>"]

        return instance

    def _pad_data_(self, tokenized_ids : list, pad_length : int):
        
        """
        args:
            tokenized_text - tensor to pad
            pad_length - the size to pad to

        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """

        diff = pad_length - len(tokenized_ids)

        padded = tokenized_ids + [0]*diff
        
        return padded[:pad_length]

