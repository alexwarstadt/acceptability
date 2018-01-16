import random
import time
import torch.nn as nn
import gflags
from torch.autograd import Variable
from models import model_trainer
from utils.classifier_utils import *

# EVALUATE_EVERY = 1000
#
# LOGS = open("logs/rnn-logs", "a")
# OUTPUT_PATH = "models/rnn_classifier"



def time_since(since):
    now = time.time()
    s = now - since
    h = s // 3600
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d:%d' % (h, m, s)


START_TIME = time.time()

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers, reduction_size):
        super(LSTMClassifier, self).__init__()
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


class LSTMPoolingClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers):
        super(LSTMPoolingClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.ih2h = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.pool2o = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input, hidden_states):
        # print(input)
        # print(hidden_states)
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
                 FLAGS,
                 model):
        self.FLAGS = FLAGS
        super(RNNTrainer, self).__init__(FLAGS, model)

    def to_string(self):
        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            self.model.to_string() + \
            "learning rate\t\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "experiment name\t\t\t" + self.FLAGS.experiment_name

    def get_batch_output(self, batch):
        hidden = self.model.init_hidden(batch.batch_size)
        input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.FLAGS.embedding_size)
        if self.FLAGS.gpu:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
            input = input.cuda()
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        outputs, hidden = self.model.forward(Variable(input), hidden)
        return outputs, hidden

