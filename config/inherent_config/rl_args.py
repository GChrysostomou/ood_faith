
__name__ = "rl"

SST = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 0.
    }
}

IMDB = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.00001,
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.01,
        "coherence": 2
    }
}

Yelp = AmazInstr = AmazDigiMu = AmazPantry = {
    "OPTIM_ARGS_" : {     ## for Adam Loss
        "lr" : 0.0001, ##0.00001
        "weight_decay" : 0.00001,
        "betas" : [0.9, 0.999],
        "amsgrad" : False,
    },
    "MODEL_ARGS_" : {     ## model args
        "sparsity":  0.0,
        "coherence": 0.0
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

