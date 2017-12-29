import random

import torch.nn as nn
from torch.autograd import Variable

import rnn_classifier
from models import model_trainer
from utils import classifier_data_utils as cdu
from utils.classifier_utils import *
# from utils.process_corpus import crop_sentences


class Classifier(nn.Module):
    def __init__(self, hidden_size, encoding_size):
        super(Classifier, self).__init__()
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
    def __init__(self,
                 FLAGS,
                 model,
                 encoder):
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

    def interact(self, sentences):
        # cropped_sentences = crop_sentences(sentences)
        batch = cdu.Batch(cropped_sentences, self.dm)
        sentence_vecs = self.get_sentence_vecs_without_stop_lstm(batch)
        outputs = self.model.forward(sentence_vecs, self.encoder)
        for s, o in zip(sentences, outputs):
            print(o[0].data, s)

