import lm_models
import torch
import torch.nn as nn

import utils.data_utils as du
from models import lm as my_lm



def generate(n, out_path, gpu):
    lm = lm_models.MyLSTM(input_size=300, hidden_size=350, output_size=20001, n_layers=1, nonlinearity=nn.Tanh())
    lm.load_state_dict(torch.load('models/model_7-14_15:33:4_110500'))
    dm = du.DataManager('/scratch/asw462/data/bnc-30', '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                        '/scratch/asw462/data/bnc-30/vocab_20000.txt', 300, crop_pad_length=30)
    mu = my_lm.ModelUtils(dm, gpu)
    out = open(out_path, "w+")
    lines = ""
    for i in range(n):
        lines += ("lm	0	*	<s> " + mu.generate_sans_probability(29, lm) + "</s>\n")
        if i % 1000 == 0 or i == n-1:
            out.write(lines)
            print(i, "lines printed")
            lines = ""
    out.close()

def generate_batch(n, batch, out_path, gpu):
    lm = lm_models.MyLSTM(input_size=300, hidden_size=350, output_size=20001, n_layers=1, nonlinearity=nn.Tanh())
    lm.load_state_dict(torch.load('models/model_7-14_15:33:4_110500'))
    dm = du.DataManager('/scratch/asw462/data/bnc-30', '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                        '/scratch/asw462/data/bnc-30/vocab_20000.txt', 300, crop_pad_length=30)
    mu = my_lm.ModelUtils(dm, gpu)
    out = open(out_path, "w+")
    lines = []
    for i in range(n):
        lines.extend(mu.generate_batch(29, batch, lm))
        if i % 1 == 0 or i == n - 1:
            lines = ["lm	0	*	<s> " + l + "</s>\n" for l in lines]
            for l in lines:
                out.write(l)
            print(i, "lines printed")
            lines = []
    out.close()


generate(1000000, "acceptability_corpus/lm_generated6", True)
