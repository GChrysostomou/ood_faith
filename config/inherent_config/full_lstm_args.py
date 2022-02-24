
__name__ = "full_lstm"



SST = IMDB = Yelp = AmazInstr = AmazDigiMu = AmazPantry = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.001, 
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "dropout":         0.1,
    }
}

get_ = {
    "SST": SST,
    "IMDB": IMDB,
    "Yelp" : Yelp,
    "AmazPantry" : AmazPantry,
    "AmazInstr" : AmazInstr,
    "AmazDigiMu" : AmazDigiMu
}

