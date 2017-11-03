import random

import torch.nn as nn
from torch.autograd import Variable

import rnn_classifier
from models import model_trainer
from utils import classifier_data_utils as cdu
from utils.classifier_utils import *
from utils.process_corpus import crop_sentences


class Classifier(nn.Module):
    def __init__(self, hidden_size, encoding_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.lm2h = nn.Linear(encoding_size, self.hidden_size)
        self.h20 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, sentence_vecs):
        hidden = self.tanh(self.lm2h(sentence_vecs))
        out = self.sigmoid(self.h20(hidden))
        return out


class AJTrainer(model_trainer.ModelTrainer):
    def __init__(self,
                 corpus_path,
                 embedding_path,
                 vocab_path,
                 embedding_size,
                 model,
                 encoder,
                 stages_per_epoch,
                 prints_per_stage,
                 convergence_threshold,
                 max_epochs,
                 gpu,
                 learning_rate=.01):
        super(AJTrainer, self).__init__(corpus_path, embedding_path, vocab_path, embedding_size, model, stages_per_epoch,
                                         prints_per_stage, convergence_threshold, max_epochs, gpu, learning_rate)
        self.encoder = encoder
        self.LOGS_PATH = "logs/aj_logs_" + self.time_stamp
        self.OUTPUT_PATH = "models/aj_classifier_" + self.time_stamp
        self.LOGS = open(self.LOGS_PATH, "a")
        self.OUT_LOGS = open("logs/aj_outputs_" + self.time_stamp, "a")
        if self.gpu:
            self.encoder = self.encoder.cuda()

    def to_string(self):
        return "data\t\t\t" + self.corpus_path + "\n" + \
            "input size\t\t" + str(self.embedding_size) + "\n" + \
            "hidden size\t\t" + str(self.model.hidden_size) + "\n" + \
            "encoder hidden\t\t" + str(self.encoder.hidden_size) + "\n" + \
            "encoder reduction\t" + str(self.encoder.reduction_size) + "\n" + \
            "dencoder num layers\t" + str(self.encoder.num_layers) + "\n" + \
            "learning rate\t\t" + str(self.learning_rate) + "\n" + \
            "output\t\t\t" + str(self.OUTPUT_PATH)

    def get_batch_output(self, batch):
        hidden = self.encoder.init_hidden(batch.batch_size)
        input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.embedding_size)
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        if self.gpu:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
            input = input.cuda()
        _, reduction = self.encoder.forward(Variable(input), hidden)
        output = self.model.forward(reduction)
        return output, None

    def interact(self, sentences):
        cropped_sentences = crop_sentences(sentences)
        batch = cdu.Batch(cropped_sentences, self.dm)
        sentence_vecs = self.get_sentence_vecs_without_stop_lstm(batch)
        outputs = self.model.forward(sentence_vecs, self.encoder)
        for s, o in zip(sentences, outputs):
            print(o[0].data, s)







# ============= LOAD ENCODER =============

encoder_path = 'models/rnn_classifier_9-27_18:12:45'
encoder = rnn_classifier.Classifier(hidden_size=306, embedding_size=300, num_layers=3, reduction_size=165)
encoder.load_state_dict(torch.load(encoder_path))

# ============= EXPERIMENT ================

def random_experiment():
    h_size = int(math.floor(math.pow(random.uniform(4, 10), 2)))  # [16, 100], quadratic distribution
    lr = math.pow(.1, random.uniform(0.5, 3))  # [.3, .001] log distribution
    cl = Classifier(hidden_size=h_size, encoding_size=encoder.reduction_size)
    clt = AJTrainer('acceptability_corpus/balanced',
                    '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                    '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                    300,
                    cl,
                    encoder,
                    stages_per_epoch=1,
                    prints_per_stage=1,
                    convergence_threshold=20,
                    max_epochs=100,
                    gpu=False,
                    learning_rate=lr)
    clt.run()

def random_experiment_local():
    h_size = int(math.floor(math.pow(random.uniform(4, 10), 2)))  # [16, 100], quadratic distribution
    lr = math.pow(.1, random.uniform(1.5, 4))  # [.01, .00001] log distribution
    cl = Classifier(hidden_size=h_size, encoding_size=encoder.reduction_size)
    clt = AJTrainer('acceptability_corpus/levin',
                    '../data/bnc-30/embeddings_20000.txt',
                    '../data/bnc-30/vocab_20000.txt',
                    300,
                    cl,
                    encoder,
                    stages_per_epoch=1,
                    prints_per_stage=1,
                    convergence_threshold=20,
                    max_epochs=100,
                    gpu=False,
                    learning_rate=lr)
    clt.run()

# random_experiment_local()

def resume_experiment(model_path, h_size, num_layers, reduction_size, lr):
    cl = Classifier(hidden_size=h_size, encoding_size=encoder.reduction_size)
    cl.load_state_dict(torch.load(model_path))
    clt = AJTrainer('acceptability_corpus',
                    '/scratch/asw462/data/bnc-30/embeddings_20000.txt',
                    '/scratch/asw462/data/bnc-30/vocab_20000.txt',
                    300,
                    cl,
                    encoder,
                    stages_per_epoch=1,
                    prints_per_stage=1,
                    convergence_threshold=20,
                    max_epochs=100,
                    gpu=False,
                    learning_rate=lr)
    clt.run()






#     def get_sentence_vecs(self, batch):
#         input = batch.tensor_view
#         hidden = self.encoder.init_hidden(batch.batch_size)
#         hiddens = []
#         for word_batch in input[:-1]:
#             output, hidden = self.encoder.forward(Variable(word_batch), hidden)
#             hiddens.append(hidden)
#         h_sum = torch.zeros(batch.batch_size, self.encoder.hidden_size)
#         for h in hiddens:
#             h_sum += h[-1][0].data
#         return h_sum / len(input)
#
#     def get_sentence_vecs_without_stop_lstm(self, batch):
#         hidden = self.encoder.init_hidden(batch.batch_size)
#         input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.embedding_size)
#         for i, t in enumerate(batch.tensor_view):
#             input[i] = t
#         _, hiddens = self.encoder.forward(Variable(input), hidden)
#         h_sum = torch.zeros(batch.batch_size, self.encoder.hidden_size)
#         sum_n = [0] * batch.batch_size
#         for b_i in range(batch.batch_size):
#             for w_i in range(batch.sentence_length - 1):
#                 try:
#                     if batch.words_view[b_i][w_i + 1] != STOP:
#                         h_sum[b_i] += hiddens[w_i][b_i].data
#                         sum_n[b_i] = sum_n[b_i] + 1
#                 except IndexError:
#                     pass
#         for b_i in range(batch.batch_size):
#             h_sum[b_i] = h_sum[b_i] / sum_n[b_i]
#         return h_sum
#
#     def get_final_sentence_vec_lstm(self, batch):
#         hidden = self.encoder.init_hidden(batch.batch_size)
#         input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.embedding_size)
#         for i, t in enumerate(batch.tensor_view):
#             input[i] = t
#         _, hiddens = self.encoder.forward(Variable(input), hidden)
#         h_sum = torch.zeros(batch.batch_size, self.encoder.hidden_size)
#         sum_n = [0] * batch.batch_size
#         for b_i in range(batch.batch_size):
#             for w_i in range(batch.sentence_length - 1):
#                 try:
#                     if batch.words_view[b_i][w_i + 1] != STOP:
#                         h_sum[b_i] += hiddens[w_i][b_i].data
#                         sum_n[b_i] = sum_n[b_i] + 1
#                 except IndexError:
#                     pass
#         for b_i in range(batch.batch_size):
#             h_sum[b_i] = h_sum[b_i] / sum_n[b_i]
#         return h_sum
#
#
#     def get_sentence_vecs_without_stop(self, batch):
#         input = batch.tensor_view
#         hidden = self.encoder.init_hidden(batch.batch_size)
#         hiddens = [] #30x36x350
#         # batch.words_view :: 36 x 31
#         for word_batch in input[:-1]:
#             output, hidden = self.encoder.forward(Variable(word_batch), hidden)
#             hiddens.append(hidden)
#         h_sum = torch.zeros(batch.batch_size, self.encoder.hidden_size)
#         sum_n = [0] * batch.batch_size
#         for b_i in range(batch.batch_size):
#             for w_i in range(batch.sentence_length-1):
#                 try:
#                     if batch.words_view[b_i][w_i+1] != STOP:
#                         h_sum[b_i] += hiddens[w_i][-1][0][b_i].data
#                         sum_n[b_i] = sum_n[b_i] + 1
#                 except IndexError:
#                     pass
#         for b_i in range(batch.batch_size):
#             h_sum[b_i] = h_sum[b_i] / sum_n[b_i]
#         return h_sum
#
#     def get_sentence_vecs_lazy(self, batch):
#         h_sum = torch.zeros(batch.batch_size, self.embedding_size)
#         for i, i_th_word_vecs in enumerate(batch.tensor_view):
#             h_sum += i_th_word_vecs
#         return h_sum / batch.sentence_length
#
#     def get_sentence_vecs_lazy_without_stop(self, batch):
#         h_sum = torch.zeros(batch.batch_size, self.embedding_size)
#         for b_i, sentence in enumerate(batch.words_view):
#             n_words = 0
#             for w in sentence:
#                 if w != STOP:
#                     n_words += 1
#                     h_sum[b_i] += self.dm.word_to_tensor(w)
#             h_sum[b_i] = h_sum[b_i] / n_words
#         return h_sum

    # def print_min_and_max(self, outputs, batch):
    #     max_prob, max_i_sentence = torch.topk(outputs.data, 1, 0)
    #     min_prob, min_i_sentence = torch.topk(outputs.data * -1, 1, 0)
    #     max_sentence = batch.sentences_view[max_i_sentence[0][0]]
    #     min_sentence = batch.sentences_view[min_i_sentence[0][0]]
    #     print("max:", max_prob[0][0], max_sentence)
    #     print("min:", min_prob[0][0] * -1, min_sentence)

    # def get_target_weights(self, targets):
    #     weights = []
    #     for t in targets:
    #         if t < .5:
    #             weights.append(1 - self.dm.corpus_bias)
    #         else:
    #             weights.append(self.dm.corpus_bias)
    #     return weights







