# Create a script to run a random hyperparameter search.

import random
import numpy as np
import gflags
import sys
from datetime import datetime

SINGLE_DAY = True
now = datetime.now()
NAME = "{0[0]:02d}{0[1]:02d}{0[2]:02d}{0[3]:02d}{0[4]:02d}".format([now.month, now.day, now.hour, now.minute, now.second])
SWEEP_RUNS = 200

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"
BOOL = "BOOL"
CHOICE = "CHOICE"
MUL = "MUL" # multiple of 100
QUAD = "QUAD"

FLAGS = gflags.FLAGS

#
# def random_experiment_local():
#     h_size = int(math.floor(math.pow(random.uniform(4, 10), 2)))  # [16, 100], quadratic distribution
#     lr = math.pow(.1, random.uniform(1.5, 4))  # [.01, .00001] log distribution
#     cl = Classifier(hidden_size=h_size, encoding_size=encoder.reduction_size)
#     clt = AJTrainer('acceptability_corpus/levin',
#                     '../data/bnc-30/embeddings_20000.txt',
#                     '../data/bnc-30/vocab_20000.txt',
#                     300,
#                     cl,
#                     encoder,
#                     stages_per_epoch=1,
#                     prints_per_stage=1,
#                     convergence_threshold=20,
#                     max_epochs=100,
#                     gpu=False,
#                     learning_rate=lr)
#     clt.run()




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
    # "data_dir": "/scratch/asw462/data/bnc_lm/",
    "vocab_path": "/scratch/asw462/data/bnc-30/vocab_20000.txt",
    "embedding_path": "/scratch/asw462/data/bnc-30/embeddings_20000.txt",
    "log_path": "/scratch/asw462/logs/",
    "data_type":     "discriminator",
    "encoder_type":    "rnn_classifier_pooling",
    "model_type":      "aj_classifier",
    "ckpt_path":  "/scratch/asw462/models/",
    "gpu": "",

    #sizes
    "embedding_size":   "300",
    "crop_pad_length": "30",


    #chunks
    "stages_per_epoch": "1",
    "prints_per_stage": "1",
    "convergence_threshold": "20",
    "max_epochs": "500",
    "batch_size": "32",


}

# Tunable parameters.
SWEEP_PARAMETERS = {

    "hidden_size": ("h_size", QUAD, 10, 200),
    "learning_rate": ("lr", EXP, 0.005, 0.00005),
    "data_dir": ("data", CHOICE, [
                                  "/scratch/asw462/data/aj_all/",
                                  "/scratch/asw462/data/aj_balanced",
                                  "/scratch/asw462/data/levin",
                                  "/scratch/asw462/data/levin_balanced"
                                  ], None),

    
"sweep_1124013946/sweep_1124013946_rnn_classifier_pooling_1-lr0.00015-h_size689-datapermuted_0-6-num_layers4",
"sweep_1124013946/sweep_1124014204_rnn_classifier_pooling_1-lr8e-05-h_size1034-datapermuted_0-6-num_layers3",
"sweep_1124013946/sweep_1124013946_rnn_classifier_pooling_2-lr0.00015-h_size689-datapermuted_2-4-num_layers4",
"sweep_1124013946/sweep_1124014204_rnn_classifier_pooling_2-lr8e-05-h_size1034-datapermuted_0-2-num_layers3",
"sweep_1124013946/sweep_1124013946_rnn_classifier_pooling_3-lr0.00015-h_size689-datapermuted_0-2-num_layers4",
"sweep_1124013946/sweep_1124014204_rnn_classifier_pooling_3-lr8e-05-h_size1034-datapermuted_4-6-num_layers3",
"sweep_1124013946/sweep_1124013946_rnn_classifier_pooling_4-lr0.00015-h_size689-datapermuted_4-6-num_layers4",
"sweep_1124013946/sweep_1124014204_rnn_classifier_pooling_4-lr8e-05-h_size1034-datapermuted_2-4-num_layers3",


"sweep_1124_early_stop_day2.5/sweep_1124013946_rnn_classifier_pooling_1-lr0.00015-h_size689-datapermuted_0-6-num_layers4",
"sweep_1124_early_stop_day2.5/sweep_1124014204_rnn_classifier_pooling_1-lr8e-05-h_size1034-datapermuted_0-6-num_layers3",
"sweep_1124_early_stop_day2.5/sweep_1124013946_rnn_classifier_pooling_2-lr0.00015-h_size689-datapermuted_2-4-num_layers4",
"sweep_1124_early_stop_day2.5/sweep_1124014204_rnn_classifier_pooling_2-lr8e-05-h_size1034-datapermuted_0-2-num_layers3",
"sweep_1124_early_stop_day2.5/sweep_1124013946_rnn_classifier_pooling_3-lr0.00015-h_size689-datapermuted_0-2-num_layers4",
"sweep_1124_early_stop_day2.5/sweep_1124014204_rnn_classifier_pooling_3-lr8e-05-h_size1034-datapermuted_4-6-num_layers3",
"sweep_1124_early_stop_day2.5/sweep_1124013946_rnn_classifier_pooling_4-lr0.00015-h_size689-datapermuted_4-6-num_layers4",
"sweep_1124_early_stop_day2.5/sweep_1124014204_rnn_classifier_pooling_4-lr8e-05-h_size1034-datapermuted_2-4-num_layers3",


"sweep_1124_early_stop_day4/sweep_1124013946_rnn_classifier_pooling_1-lr0.00015-h_size689-datapermuted_0-6-num_layers4",
"sweep_1124_early_stop_day4/sweep_1124014204_rnn_classifier_pooling_1-lr8e-05-h_size1034-datapermuted_0-6-num_layers3",
"sweep_1124_early_stop_day4/sweep_1124013946_rnn_classifier_pooling_2-lr0.00015-h_size689-datapermuted_2-4-num_layers4",
"sweep_1124_early_stop_day4/sweep_1124014204_rnn_classifier_pooling_2-lr8e-05-h_size1034-datapermuted_0-2-num_layers3",
"sweep_1124_early_stop_day4/sweep_1124013946_rnn_classifier_pooling_3-lr0.00015-h_size689-datapermuted_0-2-num_layers4",
"sweep_1124_early_stop_day4/sweep_1124014204_rnn_classifier_pooling_3-lr8e-05-h_size1034-datapermuted_4-6-num_layers3",
"sweep_1124_early_stop_day4/sweep_1124013946_rnn_classifier_pooling_4-lr0.00015-h_size689-datapermuted_4-6-num_layers4",
"sweep_1124_early_stop_day4/sweep_1124014204_rnn_classifier_pooling_4-lr8e-05-h_size1034-datapermuted_2-4-num_layers3",



"sweep_121800/sweep_1218005439_rnn_classifier_pooling_89-lr8e-05-h_size1034-dataperm-1-6-num_layers3",
"sweep_121800/sweep_1218010112_rnn_classifier_pooling_83-lr0.00015-h_size689-dataperm-3-4-num_layers4",
"sweep_121800/sweep_1218005439_rnn_classifier_pooling_95-lr8e-05-h_size1034-dataperm-1-2-num_layers3",
"sweep_121800/sweep_1218010112_rnn_classifier_pooling_91-lr0.00015-h_size689-dataperm-1-2-num_layers4",
"sweep_121800/sweep_1218005439_rnn_classifier_pooling_96-lr8e-05-h_size1034-dataperm-5-6-num_layers3",
"sweep_121800/sweep_1218010112_rnn_classifier_pooling_96-lr0.00015-h_size689-dataperm-1-6-num_layers4",
"sweep_121800/sweep_1218005439_rnn_classifier_pooling_98-lr8e-05-h_size1034-dataperm-3-4-num_layers3",
"sweep_121800/sweep_1218010112_rnn_classifier_pooling_97-lr0.00015-h_size689-dataperm-5-6-num_layers4",


"sweep_121804/sweep_1218040205_rnn_classifier_pooling_93-lr0.00015-h_size689-datashuff-10-15-num_layers4",
"sweep_121804/sweep_1218040348_rnn_classifier_pooling_94-lr8e-05-h_size1034-datashuff-10-15-num_layers3",
"sweep_121804/sweep_1218040205_rnn_classifier_pooling_96-lr0.00015-h_size689-datashuff-05-10-num_layers4",
"sweep_121804/sweep_1218040348_rnn_classifier_pooling_95-lr8e-05-h_size1034-datashuff-15-20-num_layers3",
"sweep_121804/sweep_1218040205_rnn_classifier_pooling_99-lr0.00015-h_size689-datashuff-15-20-num_layers4",
"sweep_121804/sweep_1218040348_rnn_classifier_pooling_99-lr8e-05-h_size1034-datashuff-00-05-num_layers3",
"sweep_121804/sweep_1218040348_rnn_classifier_pooling_93-lr8e-05-h_size1034-datashuff-05-10-num_layers3 ",
"sweep_121804/sweep_1218040503_rnn_classifier_pooling_92-lr8e-05-h_size689-datashuff-00-05-num_layers4"

    
    
    "encoder_path": ("enc", CHOICE,
                     [



                         "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_10-lr0.00044-h_size279-databnc_lm-num_layers2",
                        "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_14-lr0.00088-h_size748-datadiscriminator-num_layers3",
                        "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_15-lr0.0002-h_size1019-databnc_lm-num_layers2",
                        "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_16-lr8e-05-h_size1034-datadiscriminator-num_layers3",
                        "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_17-lr0.00015-h_size689-datadiscriminator-num_layers4",
                        "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_19-lr0.00029-h_size313-databnc_lm-num_layers1",
                        "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_1-lr0.00022-h_size1515-databnc_lm-num_layers1",
                        "/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_4-lr0.00015-h_size231-databnc_lm-num_layers4"
                      ], None)

}


sweep_name = "sweep_" + NAME + "_" + FIXED_PARAMETERS["model_type"]

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(SWEEP_RUNS)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print
print "SWEEP=\"%s\" " % NAME
print "mkdir ~/logs/$SWEEP"
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
            if param == "encoder_path":
                params["encoding_size"] = \
                    int(filter(lambda s: s.startswith("h_size"), sample.split("-"))[0].replace("h_size", ""))
                params["encoder_num_layers"] = \
                    int(filter(lambda s: s.startswith("num_layers"), sample.split("-"))[0].replace("num_layers", ""))
        elif t==MUL:
            x = mn + (mx - mn) * r
            sample = x + 100 - x % 100
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
        if "/" in val_disp:
            val_disp = filter(lambda x: x != "", val_disp.split("/"))[-1]
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
        print "FLAGS=\"" + flags + "\" sbatch ~/scripts/train_rnn.sbatch"
    else:
        print "MODEL=\"rnn_classifier\" FLAGS=\"" + flags + "\" bash ~/scripts/sbatch_submit.sh"
    print


