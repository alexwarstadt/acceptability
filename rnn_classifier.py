import torch
import torch.nn as nn
import model
import lm as my_lm
import math
import data_utils as du
import random
import classifier_data_utils as cdu
import time
import model_trainer
from constants import *
from classifier_utils import *
from torch.autograd import Variable

# EVALUATE_EVERY = 1000

LOGS = open("logs/rnn-logs", "a")
OUTPUT_PATH = "models/rnn_classifier"

def time_since(since):
    now = time.time()
    s = now - since
    h = s // 3600
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d:%d' % (h, m, s)


START_TIME = time.time()

class Classifier(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.ih2h = nn.LSTM(embedding_size, hidden_size, bidirectional=True)
        self.h2o = nn.Linear(2 * hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input, hidden_states):
        o, hc = self.ih2h(input, hidden_states)
        output = self.sigmoid(self.h2o(o[-1]))
        return output, o

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(2, batch_size, self.hidden_size)),
                Variable(torch.zeros(2, batch_size, self.hidden_size)))

class RNNTrainer(model_trainer.ModelTrainer):
    def __init__(self,
                 corpus_path,
                 embedding_path,
                 vocab_path,
                 embedding_size,
                 model,
                 stages_per_epoch,
                 prints_per_stage,
                 convergence_threshold,
                 max_epochs,
                 gpu,
                 learning_rate=.01):
        super(RNNTrainer, self).__init__(corpus_path, embedding_path, vocab_path, embedding_size, model, stages_per_epoch,
                                         prints_per_stage, convergence_threshold, max_epochs, gpu, learning_rate)


#============= EXPERIMENT ================
size_range = (100, 500)
lr = (1, 4)

for _ in range(10):
    cl = Classifier(random.randint(size_range[0], size_range[1]), 300)
    clt = RNNTrainer('../data/discriminator/',
                     '../data/bnc-30/embeddings_20000.txt',
                     '../data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=10,
                     prints_per_stage=100,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=False,
                     learning_rate=math.pow(.2, random.uniform(lr[0], lr[1])))
    clt.run()