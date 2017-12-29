import random

import torch.nn as nn
from torch.autograd import Variable

import rnn_classifier
from models import model_trainer
from utils import classifier_data_utils as cdu
from utils.classifier_utils import *
# from utils.process_corpus import crop_sentences


class Classifier(nn.Module):
    def __init__(self, hidden_size, input_size, max_pool):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.max_pool = max_pool
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        if self.max_pool:
            encoding = nn.functional.max_pool1d(torch.transpose(inputs, 0, 2), len(inputs))
            encoding = torch.transpose(encoding.squeeze(), 0, 1)
        else:
            encoding = inputs.sum(0)
        hidden = self.tanh(self.i2h(encoding))
        out = self.sigmoid(self.h2o(hidden))
        return out



class CbowTrainer(model_trainer.ModelTrainer):
    def __init__(self,
                 FLAGS,
                 model,):
        super(CbowTrainer, self).__init__(FLAGS, model)
        if FLAGS.gpu:
            self.encoder = self.encoder.cuda()

    def to_string(self):

        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            "input size\t\t" + str(self.FLAGS.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.FLAGS.hidden_size) + "\n" + \
            "learning rate\t\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "max pool\t\t\t" + str(self.model.max_pool) + "\n" + \
            "experiment name\t\t" + self.FLAGS.experiment_name

    def get_batch_output(self, batch):
        input = Variable(torch.Tensor(len(batch.tensor_view), batch.batch_size, self.FLAGS.embedding_size))
        # print("batch.tensor_view!", batch.tensor_view)
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        if self.FLAGS.gpu:
            input = input.cuda()
        output = self.model.forward(input)
        return output, None


