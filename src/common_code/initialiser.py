import json
import glob
import config
import os
  
def prepare_config(user_args, stage = "train") -> dict:

  # passed arguments
  dataset = user_args["dataset"]

  with open('config/model_config.json', 'r') as f:
      default_args = json.load(f)

  if stage == "retrain":

    user_args["data_dir"] = user_args["extracted_rationale_dir"]

  data_dir = os.path.join(
    os.getcwd(), 
    user_args["data_dir"], 
    user_args["dataset"],
    "data",
    ""
  )

  ## activated when training on rationales
  if "model_dir" not in user_args: model_dir = user_args["rationale_model_dir"]
  else: model_dir = user_args["model_dir"]

  model_dir = os.path.join(
    os.getcwd(), 
    model_dir, 
    user_args["dataset"],
    ""
  )

  if stage == "extract":

    extract_dir = os.path.join(
        os.getcwd(), 
        user_args["extracted_rationale_dir"], 
        user_args["dataset"],
        ""
    )
  
  else: extract_dir = None


  if stage == "evaluate":

    ## in case we haven't run extract first
    extract_dir = os.path.join(
        os.getcwd(), 
        user_args["extracted_rationale_dir"], 
        user_args["dataset"],
        ""
    )

    eval_dir = os.path.join(
        os.getcwd(), 
        user_args["evaluation_dir"], 
        user_args["dataset"],
        ""
    )
  
  else: eval_dir = None


  # if user_args["dataset"] in {"SST", "IMDB"}: query = False
  # else: query = True
  query = False

  if stage == "evaluate" or stage == "extract": user_args["seed"] = None

  if "inherently_faithful" not in user_args: user_args["inherently_faithful"] = False

  if user_args["inherently_faithful"]:

    model_abbrev = f"{user_args['inherently_faithful']}-" + default_args["model_abbreviation"][default_args[user_args["dataset"]]["model"]]
    epochs = 20 if user_args["dataset"] != "Yelp" else 10
    embed_model = default_args["embed_model"]

    if user_args["inherently_faithful"] == "full_lstm":

      epochs = 5

  else:

    model_abbrev = default_args["model_abbreviation"][default_args[user_args["dataset"]]["model"]] 
    epochs = default_args["epochs"]
    embed_model = None

  if user_args["use_tasc"]:

    model_abbrev = "tasc_" + model_abbrev
    epochs = 5

  ood_dataset_1 = default_args[user_args["dataset"]]["ood_dataset_1"]
  ood_dataset_2 = default_args[user_args["dataset"]]["ood_dataset_2"]

  ood_rat_1 = default_args[ood_dataset_1]["rationale_length"]
  ood_rat_2 = default_args[ood_dataset_2]["rationale_length"]

  comb_args = dict(user_args, **default_args[user_args["dataset"]], **{

            "seed":user_args["seed"], 
            "epochs":epochs,
            "data_dir" : data_dir, 
            "model_abbreviation": model_abbrev,
            "model_dir": model_dir,
            "evaluation_dir": eval_dir,
            "extracted_rationale_dir": extract_dir,
            "query": query,
            "stage_of_proj" : stage,
            "ood_rat_1" : ood_rat_1, 
            "ood_rat_2" : ood_rat_2,
            "embed_model" : embed_model
  })

  #### saving config file for this run
  with open(config.cfg.config_directory + 'instance_config.json', 'w') as file:
      file.write(json.dumps(comb_args,  indent=4, sort_keys=True))

  return comb_args

def make_folders(args, stage):

  assert stage in ["train", "extract", "retrain", "evaluate"]

  if stage == "train":

    os.makedirs(args["model_dir"] + "/model_run_stats/", exist_ok=True)
    print("\nFull text models saved in: {}\n".format(args["model_dir"]))

  if stage == "evaluate":

    os.makedirs(args["evaluation_dir"], exist_ok=True)
    print("\nFaithfulness results saved in: {}\n".format(args["evaluation_dir"]))

  if stage == "extract":

    os.makedirs(args["extracted_rationale_dir"], exist_ok=True)
    print("\nExtracted rationales saved in: {}\n".format(args["extracted_rationale_dir"]))

  
  if stage == "retrain":

    os.makedirs(os.path.join(args["model_dir"],args["thresholder"]) + "/model_run_stats/", exist_ok=True)
    print("\nRationale models saved in: {}\n".format(os.path.join(args["model_dir"],args["thresholder"])))


  return

def initial_preparations(user_args, stage):

    comb_args = prepare_config(
      user_args, 
      stage)

    make_folders(
      comb_args, 
      stage
      )

    return comb_args
