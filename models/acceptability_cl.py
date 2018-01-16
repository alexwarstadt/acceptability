import random

import torch.nn as nn
from torch.autograd import Variable

import rnn_classifier
from models import model_trainer
from utils import classifier_data_utils as cdu
from utils.classifier_utils import *
# from utils.process_corpus import crop_sentences


class AcceptabilityClassifier(nn.Module):
    """
    A basic linear classifier for acceptability judgments.
    Input sentence embedding (with size = 2*encoding_size, factor of 2 comes from bidirectional LSTM encoding)
    Hidden layer (encoding 2*encoding_size X hidden_size)
    Output layer (hidden_size X 1)
    """
    def __init__(self, hidden_size, encoding_size):
        super(AcceptabilityClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.enc2h = nn.Linear(2*encoding_size, self.hidden_size)
        self.h20 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, sentence_vecs):
        hidden = self.tanh(self.enc2h(sentence_vecs))
        out = self.sigmoid(self.h20(hidden))
        return out



class AJTrainer(model_trainer.ModelTrainer):
    """
    Extends ModelTrainer for training AJ_Classifier models
    """
    def __init__(self,
                 FLAGS,
                 model,
                 encoder):
        """
        :param FLAGS: flags are passed in by training.run_training.py
        :param model: an instance of AcceptabilityClassifier
        :param encoder: an instance of LSTMPoolingClassifier or other encoder model
        """
        super(AJTrainer, self).__init__(FLAGS, model)
        self.encoder = encoder
        if FLAGS.gpu:
            self.encoder = self.encoder.cuda()

    def to_string(self):
        return "data\t\t\t" + self.FLAGS.data_dir + "\n" + \
            "input size\t\t" + str(self.FLAGS.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.FLAGS.hidden_size) + "\n" + \
            "learning rate\t\t" + str(self.FLAGS.learning_rate) + "\n" + \
            "encoding size\t\t" + str(self.FLAGS.encoding_size) + "\n" + \
            "encoder name\t\t" + str(self.FLAGS.encoder_path) + "\n" + \
            "experiment name\t\t" + self.FLAGS.experiment_name

    def get_batch_output(self, batch):
        hidden = self.encoder.init_hidden(batch.batch_size)
        input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.FLAGS.embedding_size)
        # print("batch.tensor_view!", batch.tensor_view)
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        if self.FLAGS.gpu:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
            input = input.cuda()
        _, encoding = self.encoder.forward(Variable(input), hidden)
        output = self.model.forward(encoding)
        return output, None


