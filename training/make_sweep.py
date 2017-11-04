# Create a script to run a random hyperparameter search.

import random
import numpy as np
import gflags
import sys

SINGLE_DAY = False
NAME = "10_30"
SWEEP_RUNS = 8

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"
BOOL = "BOOL"
CHOICE = "CHOICE"
MUL = "MUL" # multiple of 100
QUAD = "QUAD"

FLAGS = gflags.FLAGS


gflags.DEFINE_string("data_dir", "/scratch/asw462/data/bnc_lm/", "")
gflags.DEFINE_string("vocab_path", "/scratch/asw462/data/bnc-30/vocab_20000.txt", "")
gflags.DEFINE_string("embedding_path", "/scratch/asw462/data/bnc-30/embeddings_20000.txt", "")
gflags.DEFINE_string("log_path", "/scratch/asw462/logs/", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    #paths
    "data_dir": "/scratch/asw462/data/bnc_lm/",
    "vocab_path": "/scratch/asw462/data/bnc-30/vocab_20000.txt",
    "embedding_path": "/scratch/asw462/data/bnc-30/embeddings_20000.txt",
    "log_path": "/scratch/asw462/logs/",
    "data_type":     "discriminator",
    "model_type":      "rnn_classifier_pooling",
    "ckpt_path":  "/scratch/asw462/models/",
    "gpu": "True",

    #sizes
    "embedding_size":   "300",
    "crop_pad_length": "30",


    #chunks
    "stages_per_epoch": "100",
    "prints_per_stage": "1",
    "convergence_threshold": "20",
    "batch_size": "32",


}

# Tunable parameters.
SWEEP_PARAMETERS = {

    "hidden_size": ("h_size", QUAD, 225, 1600),
    "num_layers": ("num_layers", LIN, 1, 4),
    "learning_rate": ("lr", EXP, 0.00005, 0.005),

}


sweep_name = "sweep_" + NAME + "_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(SWEEP_RUNS)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

for run_id in range(SWEEP_RUNS):
    params = {}
    name = sweep_name + "_" + str(run_id)

    params.update(FIXED_PARAMETERS)
    # Any param appearing in both sets will be overwritten by the sweep value.

    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[1]
        mn = config[2]
        mx = config[3]

        r = random.uniform(0, 1)
        if t == EXP:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = np.exp(lmn + (lmx - lmn) * r)
        elif t == QUAD:
            sqmn = np.sqrt(mn)
            sqmx = np.sqrt(mx)
            sample = np.power(sqmn + (sqmx - sqmn) * r, 2)
        elif t == BOOL:
            sample = r > 0.5
        elif t==SS_BASE:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = 1 - np.exp(lmn + (lmx - lmn) * r)
        elif t==CHOICE:
            sample = random.choice(mn)
        elif t==MUL:
            x = mn + (mx - mn) * r
            sample =  x + 100 - x % 100
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))
            val_disp = str(sample)
            params[param] = sample
        elif isinstance(mn, float):
            val_disp = "%.2g" % sample
            params[param] = sample
        elif t==BOOL:
            val_disp = str(int(sample))
            if not sample:
                params['no' + param] = ''
            else:
                params[param] = ''
        else:
            val_disp = sample
            params[param] = sample
        name += "-" + config[0] + val_disp

    flags = ""
    for param in params:
        value = params[param]
        flags += " --" + param + " " + str(value)
        if param == "es_num_roots":
            root = value
        elif param == "es_num_episodes":
            eps = value

    flags += " --experiment_name " + name

    if SINGLE_DAY:
        print "cd ~/acceptability; python -m rnn_classifier " + flags
    else:
        print "MODEL=\"rnn_classifier\" FLAGS=\"" + flags + "\" bash ~/scripts/sbatch_submit.sh"
    print
