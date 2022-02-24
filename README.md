Repository for the paper *"An Empirical Study on Explanations in Out-of-Domain Settings"* to appear at ACL 2022. 

## Prerequisites

Install necessary packages by using the files  [conda_reqs.txt]() and  [pip_reqs.txt]()

```
conda create --name ood_faith --file  conda_reqs.txt
conda activate ood_faith
pip install -r pip_reqs.txt
python -m spacy download en
```

## Downloading Task Data
Run the following script from this directory:

```
python src/data_functions/data_processors.py --data_directory "datasets"
```

This script downloads temporary data related to our datasets, and processes them and saves them in a json format in *--data_directory*. Also generates a description of the data splits with related data statistics.

## Training Models

You can train the following models with [train_fulltext_and_kuma.py]() script:

(1) BERT-base on full-text (default option --> inherently_faithful == None)

(2) bi-LSTM on full-text (inherently_faithful == "full_lstm")

(3) HardKuma models (inherently_faithful == "kuma")

(4) Lei et. al. models (inherently_faithful == "rl")

, using the following options:

* dataset : *{"SST","IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr"}*

* data_dir : *directory where task data is* 

* model_dir : *directory for saving trained models*

* seed : *random seed for the experiment*

* evaluate_models : *used for evaluating trained models on test set*

* inherently_faithful : *{"kuma", "rl", "full_lstm", None}*

  

  Example script:

``` shell
for seed in 5 10 15 20 25
do	
    python train_fulltext_and_kuma.py 
                                    --dataset SST 
                                    --data_dir data/ 
                                    --model_dir models/ 
                                    --seed $seed
done    
python train_fulltext_and_kuma.py 
                            --dataset SST 
                            --data_dir data/ 
                            --model_dir models/ 
                            --evaluate_models
```

## Evaluating post-hoc explanation faithfulness 

You can run sufficiency and comprehensiveness tests using the  [evaluate_posthoc.py]() script, using the following options:

* dataset : *{"SST","IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr"}*

* data_dir : *directory where task data is* 

* model_dir : *directory for saving trained models*

* evaluation_dir : *directory for saving faithfulness results*

* thresholder : *{"topk", "contigious"}*

* inherently_faithful : *{None}*

  

  Example script:

```shell
python evaluate_posthoc.py 
	    --dataset SST 
	    --data_dir data/ 
	    --model_dir models/ 
	    --evaluation_dir posthoc_results/
	    --thresholder "topk" 
```



## Extracting rationales for FRESH

You can extract rationales from all feature attributions using the [FRESH_extract_rationales.py]() script, using the following options:

* dataset : *{"SST","IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr"}*

* data_dir : *directory where task data is* 

* model_dir : *directory for saving trained models*

* extracted_rationale_dir : *directory to save extracted_rationales*

* thresholder : *{"topk", "contigious"}*

  

  Example script:

  ```shell
  python evaluate_posthoc.py 
  	    --dataset SST 
  	    --data_dir data/ 
  	    --model_dir models/ 
  	    --extracted_rationale_dir extracted_rationales/
  	    --thresholder "topk" 
  ```

  

## Training FRESH classifier

You can train a Bert-base classifier on the rationales with [FRESH_train_on_rationales.py]() script, using the following options:

* dataset : *{"SST","IMDB", "Yelp", "AmazDigiMu", "AmazPantry", "AmazInstr"}*
* extracted_rationale_dir : *directory where extracted rationales are* 
* rationale_model_dir : *directory for saving trained FRESH classifier*
* seed : *random seed for the experiment*
* evaluate_models : *used for evaluating trained models on test set*
* importance_metric : *{"attention", "gradients", "scaled attention", "ig", "deeplift"}*
* thresholder : *{"topk", "contigious"}*



Example script:

```shell
feature_attribution="scaled attention"

for seed in 5 10 15 20 25
do	
    python FRESH_train_on_rationales.py 
                    --dataset SST 
                    --extracted_rationale_dir extracted_rationales/ 
                    --rationale_model_dir FRESH_classifiers/ 
                    --thresholder "topk"
                    --seed $seed
                    --importance_metric $feature_attribution
done    
python FRESH_train_on_rationales.py 
                    --dataset SST 
                    --data_dir data/ 
                    --model_dir models/ 
                    --extracted_rationale_dir extracted_rationales/ 
                    --rationale_model_dir FRESH_classifiers/ 
                    --thresholder "topk"
                    --seed $seed
                    --importance_metric $feature_attribution
                    --evaluate_models
```

