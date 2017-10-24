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
    def __init__(self, hidden_size, embedding_size, num_layers, reduction_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.reduction_size = reduction_size
        self.ih2h = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.h2r = nn.Linear(2 * hidden_size, reduction_size)
        self.r2o = nn.Linear(reduction_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input, hidden_states):
        o, hc = self.ih2h(input, hidden_states)
        reduction = self.sigmoid(self.h2r(o[-1]))
        output = self.sigmoid(self.r2o(reduction))
        return output, reduction

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)))

    def to_string(self):
        return "input size\t\t" + str(self.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.hidden_size) + "\n" + \
            "reduction size\t\t" + str(self.reduction_size) + "\n" + \
            "num layers\t\t" + str(self.num_layers) + "\n"


class ClassifierPooling(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers):
        super(ClassifierPooling, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.ih2h = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.pool2o = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input, hidden_states):
        o, hc = self.ih2h(input, hidden_states)
        pool = nn.functional.max_pool1d(torch.transpose(o, 0, 2), len(input))
        pool = torch.transpose(pool, 0, 2).squeeze()
        output = self.sigmoid(self.pool2o(pool))
        return output, pool

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size)))


    def to_string(self):
        return "input size\t\t" + str(self.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.hidden_size) + "\n" + \
            "num layers\t\t" + str(self.num_layers) + "\n"

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
        self.LOGS_PATH = "logs/rnn_logs_" + self.time_stamp
        self.OUTPUT_PATH = "models/rnn_classifier_" + self.time_stamp
        self.LOGS = open(self.LOGS_PATH, "a")
        self.OUT_LOGS = open("logs/rnn_outputs_" + self.time_stamp, "a")

    def to_string(self):
        return "data\t\t\t" + self.corpus_path + "\n" + \
            self.model.to_string() + \
            "learning rate\t\t" + str(self.learning_rate) + "\n" + \
            "output\t\t\t" + str(self.OUTPUT_PATH)

    def get_batch_output(self, batch):
        hidden = self.model.init_hidden(batch.batch_size)
        input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.embedding_size)
        if self.gpu:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
            input = input.cuda()
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        outputs, hidden = self.model.forward(Variable(input), hidden)
        return outputs, hidden


#============= EXPERIMENT ================



def random_experiment():
    h_size = int(math.floor(math.pow(random.uniform(10, 32), 2)))           # [100, 1024], quadratic distribution
    num_layers = random.randint(1, 5)
    reduction_size = int(math.floor(math.pow(random.uniform(7, 18), 2)))    # [49, 324], quadratic distribution
    lr = math.pow(.1, random.uniform(3, 4.5))                               # [.001, 3E-5], logarithmic distribution
    cl = Classifier(hidden_size=h_size, embedding_size=300, num_layers=num_layers, reduction_size=reduction_size)
    clt = RNNTrainer('/scratch/asw462/data/discriminator/',
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=True,
                     learning_rate=lr)
    clt.run()

def random_experiment_pooling():
    h_size = int(math.floor(math.pow(random.uniform(15, 40), 2)))  # [225, 1600], quadratic distribution
    num_layers = random.randint(1, 3)
    reduction_size = int(math.floor(math.pow(random.uniform(7, 18), 2)))  # [49, 324], quadratic distribution
    lr = math.pow(.1, random.uniform(3, 4.5))  # [.001, 3E-5], logarithmic distribution
    cl = ClassifierPooling(hidden_size=h_size, embedding_size=300, num_layers=num_layers)
    clt = RNNTrainer('/scratch/asw462/data/bnc_lm/',
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=True,
                     learning_rate=lr)
    clt.run()

def random_local_experiment_pooling():
    h_size = int(math.floor(math.pow(random.uniform(10, 32), 2)))  # [100, 1024], quadratic distribution
    num_layers = random.randint(1, 5)
    lr = math.pow(.1, random.uniform(3.5, 5.5))  # [.001, 3E-5], logarithmic distribution
    cl = ClassifierPooling(hidden_size=h_size, embedding_size=300, num_layers=num_layers)
    clt = RNNTrainer('../data/discriminator/',
                     '../data/bnc-30/embeddings_20000.txt',
                    '../data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=False,
                     learning_rate=lr)
    clt.run()

def resume_experiment(model_path, h_size, num_layers, reduction_size, lr):
    cl = Classifier(hidden_size=h_size, embedding_size=300, num_layers=num_layers, reduction_size=reduction_size)
    cl.load_state_dict(torch.load(model_path))
    clt = RNNTrainer('/scratch/asw462/data/discriminator/',
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=False,
                     learning_rate=lr)
    clt.run()

def resume_experiment_pooling(model_path, h_size, num_layers, lr):
    cl = ClassifierPooling(hidden_size=h_size, embedding_size=300, num_layers=num_layers)
    cl.load_state_dict(torch.load(model_path))
    clt = RNNTrainer('/scratch/asw462/data/bnc_lm/',
                     '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                     '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                     300,
                     cl,
                     stages_per_epoch=100,
                     prints_per_stage=1,
                     convergence_threshold=20,
                     max_epochs=100,
                     gpu=False,
                     learning_rate=lr)
    clt.run()


# random_local_experiment_pooling()