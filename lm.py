import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import random
from functools import reduce
from constants import *
# import matplotlib.pyplot as plt
import numpy as np
import data_utils as du
import model as model

# TODO: split into modules(?)
# TODO: keep track of training stats better
# TODO: allow for larger data sets
# TODO: split into modules(?)
# TODO: print at better intervals

# batch_size = 36
n_epochs = 20
OUTPUT_PATH = "models/model"
LOGS_PATH = "logs/logs"
ALL_LOGS = "logs/all-logs_temp"

LOGS = open("logs/all-logs", "a")
# DATA_DIR = "data/penn"
# CROP_PAD_LENGTH = 20
# embeddings_path = "glove.6B.300d.txt"
# EMBEDDING_SIZE = 300
EVALUATE_EVERY = 500
MINI_VALID_SIZE = 50
STOP_TRAINING_THRESHOLD = 20    # number of evaluations on which validation does not improve



class ModelUtils:
    def __init__(self, dm):
        self.dm = dm

    @staticmethod
    def log_prob_to_prob(output):
        prob_matrix = torch.zeros(output.size())
        for i in range(output.size()[0]):
            prob_matrix[i] = np.exp(output.data[i])
        return prob_matrix

    @staticmethod
    def perplexity(N, prob_sum):
        return math.exp((-1 * prob_sum) / N)

    @staticmethod
    def sample(weights, prob_sum):
        threshold = random.uniform(0, prob_sum)
        i = 0
        while threshold > 0 and i < len(weights):
            threshold -= weights[i]
            i += 1
        return i - 1, weights[i - 1]

    def remove_start(self, probs):
        i_start = self.dm.vocab.index(START)
        start_prob = probs[i_start]
        probs[i_start] = 0
        return probs, 1 - start_prob

    def remove_start_unk(self, probs):
        i_start = self.dm.vocab.index(START)
        start_prob = probs[i_start]
        probs[i_start] = 0
        i_unk = self.dm.vocab.index(UNK)
        unk_prob = probs[i_unk]
        probs[i_unk] = 0
        return probs, 1 - start_prob - unk_prob


    def generate(self, n, model):
        input = torch.Tensor(1, self.dm.embedding_size)
        input[0] = self.dm.word_to_tensor(START)
        hiddens = model.init_hidden_single()
        sentence_gen = ""
        for _ in range(n):
            output, hiddens = model.forward(Variable(input), hiddens)
            probs, prob_sum = self.remove_start(self.log_prob_to_prob(output)[0].tolist())
            i_choice, w_choice = self.sample(probs, prob_sum)
            word_gen = self.dm.vocab[i_choice]
            sentence_gen += word_gen + "(" + "{:.5f}".format(w_choice) + ")" + " "
            input[0] = self.dm.word_to_tensor(word_gen)
        return sentence_gen

    def generate_sans_probability(self, n, model):
        input = torch.Tensor(1, self.dm.embedding_size)
        input[0] = self.dm.word_to_tensor(START)
        hiddens = model.init_hidden_single()
        sentence_gen = ""
        for _ in range(n):
            output, hiddens = model.forward(Variable(input), hiddens)
            probs, prob_sum = self.remove_start(self.log_prob_to_prob(output)[0].tolist())
            i_choice, w_choice = self.sample(probs, prob_sum)
            word_gen = self.dm.vocab[i_choice]
            sentence_gen += word_gen + " "
            input[0] = self.dm.word_to_tensor(word_gen)
        return sentence_gen

    # def generate_sans_probability(self, n, model):
    #     input = torch.Tensor(1, self.dm.embedding_size)
    #     input[0] = self.dm.word_to_tensor(START)
    #     hiddens = model.init_hidden_single()
    #     sentence_gen = ""
    #     for _ in range(n):
    #         output, hiddens = model.forward(Variable(input), hiddens)
    #         probs, prob_sum = self.remove_start(self.log_prob_to_prob(output)[0].tolist())
    #         i_choice, w_choice = self.sample(probs, prob_sum)
    #         word_gen = self.dm.vocab[i_choice]
    #         sentence_gen += word_gen + " "
    #         input[0] = self.dm.word_to_tensor(word_gen)
    #     return sentence_gen


    def generate_batch(self, n, batch_size, model):
        input = torch.Tensor(batch_size, self.dm.embedding_size)
        for i in range(batch_size):
            input[i] = self.dm.word_to_tensor(START)
        hiddens = model.init_hidden(batch_size)
        sentence_gen = ["" for _ in range(batch_size)]
        for _ in range(n):
            output, hiddens = model.forward(Variable(input), hiddens)
            for i, o in enumerate(output):
                probs, prob_sum = self.remove_start(self.log_prob_to_prob(o).tolist())
                i_choice, w_choice = self.sample(probs, prob_sum)
                word_gen = self.dm.vocab[i_choice]
                sentence_gen[i] += word_gen + " "
                input[i] = self.dm.word_to_tensor(word_gen)
        return sentence_gen


    def generate_max(self, n, model):
        input = torch.Tensor(1, self.dm.embedding_size)

        # TODO this was changed, is it correct?
        word_gen = self.dm.vocab[random.randint(0, self.dm.n_vocab)]
        input[0] = self.dm.word_to_tensor(word_gen)

        hidden = model.init_hidden_single()
        sentence_gen = word_gen + " "
        for _ in range(n):
            output, hidden = model.forward(Variable(input), hidden)
            probs, word_indices = output.data[0].topk(2)
            word_gen = self.dm.vocab[word_indices[0]]
            if word_gen is START:
                word_gen = self.dm.vocab[word_indices[1]]
            sentence_gen += word_gen + "(" + "{:.5f}".format(math.exp(probs[0])) + ")" + " "
            input[0] = self.dm.word_to_tensor(word_gen)
        return sentence_gen

    def decode(self, outputs, batch):
        decoded_words = []
        decoded_pairs = []
        for output in outputs:
            _, word_index = output.data[0].topk(1)
            decoded_words.append(self.dm.vocab[word_index[0]])
        for j, o_t in enumerate(batch.words_view[0][1:]):
            if decoded_words[j] == o_t:
                decoded_pairs.append(o_t)
            else:
                decoded_pairs.append(decoded_words[j] + "(" + o_t + ")")
        return decoded_pairs


def time_since(since):
    now = time.time()
    s = now - since
    h = s // 3600
    s -= h * 3600
    m = math.floor(s / 60)
    s -= m * 60
    return '%d:%d:%d' % (h, m, s)


START_TIME = time.time()


class ModelTrainer:
    def __init__(self, raw_corpus, embedding_path, embeddings_size, crop_pad_length, unked,
                 dim_hidden,
                 learning_rate, model_type="RNN", n_layers=1,
                 **criterion_nonlinearity_optimizer):
        now = time.localtime()
        self.dm = du.DataManager(raw_corpus, embedding_path, embeddings_size, crop_pad_length, unked)
        self.model_utils = ModelUtils(self.dm)
        time_stamp = str(now.tm_mon) + "-" + str(now.tm_mday) + "_" \
                     + str(now.tm_hour) + ":" + str(now.tm_min) + ":" + str(now.tm_sec)
        self.output_path = OUTPUT_PATH + "_" + time_stamp
        self.logs_path = LOGS_PATH + "_" + time_stamp
        self.nonlinearity = nn.Tanh()
        self.model_type = model_type
        if model_type is "LSTM":
            self.model = model.MyLSTM(input_size=embeddings_size, hidden_size=dim_hidden, output_size=self.dm.n_vocab,
                                      n_layers=n_layers, nonlinearity=self.nonlinearity)
        else:
            self.model = model.RNN(input_size=embeddings_size, hidden_size=dim_hidden, output_size=self.dm.n_vocab,
                                   n_layers=n_layers, nonlinearity=self.nonlinearity, model_type=model_type)
        self.learning_rate = learning_rate
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)  # or use RMSProp
        self.training_losses = []
        self.training_prplxs = []
        self.valid_prplxs = []
        self.prplx_min = math.inf
        # self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, batch):
        outputs = self.get_outputs(batch)
        output_targets = self.get_output_targets(batch)
        loss = self.get_batch_loss(outputs, output_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return outputs, loss.data[0]

    def log_probability(self, batch):
        outputs = self.get_outputs(batch)
        output_targets = self.get_output_targets(batch)
        log_prob = 0
        n_probs = 0
        for i in range(len(outputs)):
            for j in range(outputs[0].size()[0]):
                #TODO only add prob if not part of stop pad
                target = output_targets[i][j].data[0]
                log_prob += outputs[i][j][target]
                n_probs += 1
                if self.dm.vocab[target] == STOP:
                    break
        return log_prob, n_probs, outputs

    def get_outputs(self, batch):
        input = batch.tensor_view
        hidden = self.model.init_hidden(batch.batch_size)
        outputs = []
        for word_batch in input[:-1]:
            output, hidden = self.model(Variable(word_batch), hidden)
            outputs.append(output)
        return outputs

    def get_hiddens(self, batch):
        input = batch.tensor_view
        hidden = self.model.init_hidden(batch.batch_size)
        hiddens = []
        for word_batch in input[:-1]:
            output, hidden = self.model(Variable(word_batch), hidden)
            hiddens.append(hidden)
        return hiddens

    def get_output_targets(self, batch):
        """returns vocab indices in sentence-length x batch size array"""
        words = batch.words_view
        output_targets = []
        for i in range(1, len(words[0])):
            word_i_targets = []
            for sentence in words:
                # try:
                if i >= len(sentence):
                    print("anomalous sentence!!", sentence)
                word = sentence[i] if sentence[i] in self.dm.vocab else "<unk>"
                word_code = self.dm.vocab.index(word)
                # except IndexError:
                #     continue
                word_i_targets.append(Variable(torch.LongTensor([word_code])))
            output_targets.append(word_i_targets)
        return output_targets

    def get_batch_loss(self, outputs, output_targets):
        loss = Variable(torch.Tensor([0]))
        for i_w, ith_outputs in enumerate(outputs):
            for i_s, out in enumerate(ith_outputs):
                output_loss = self.criterion(out, output_targets[i_w][i_s])
                loss += output_loss
        return loss

    def validate(self):
        total_log_prob = 0
        total_words = 0
        valid_epoch = du.CorpusEpoch(self.dm.valid, self.dm)
        still_going = True
        while still_going:
            batch, still_going = valid_epoch.get_next_batch()
            log_prob, n_words, outputs = self.log_probability(batch)
            total_log_prob += log_prob[0].data[0]
            total_words += n_words
        prplx = ModelUtils.perplexity(total_words, total_log_prob)
        print("valid prplx\t\t" + str(prplx))
        return prplx

    def mini_validate(self):
        total_log_prob = 0
        total_words = 0
        valid_epoch = du.CorpusEpoch(self.dm.valid, self.dm)
        still_going = True
        total_loss = 0
        # n_batches = 100 TODO change back
        for _ in range(MINI_VALID_SIZE):
            batch, still_going = valid_epoch.get_next_batch()
            log_prob, n_words, outputs = self.log_probability(batch)
            total_log_prob += log_prob[0].data[0]
            total_words += n_words
            output_targets = self.get_output_targets(batch)
            total_loss += self.get_batch_loss(outputs, output_targets).data[0]

        prplx = ModelUtils.perplexity(total_words, total_log_prob)
        # print("mini prplx\t" + str(prplx))
        return prplx, total_loss/MINI_VALID_SIZE

    def training_logs(self, outputs, batch, model):
        decoded_pairs = reduce(lambda x, y: x + " " + y, self.model_utils.decode(outputs, batch), "").strip()
        target_sentence = ""
        for word in batch.words_view[0]:
            target_sentence += word + " "
        print("sentence\t\t", target_sentence)
        print("decoded\t\t\t", decoded_pairs)
        print("generate\t\t", self.model_utils.generate(20, model))
        print("generate max\t", self.model_utils.generate_max(20, model))
        return

    def to_string(self):
        "model type\t\t" + str(self.model_type) + "\n" + \
        "input size\t\t" + str(self.model.input_size) + "\n" + \
        "hidden size\t\t" + str(self.model.hidden_size) + "\n" + \
        "output size\t\t" + str(self.model.output_size) + "\n" + \
        "learning rate\t" + str(self.learning_rate) + "\n" + \
        "# layers\t\t" + str(self.model.n_layers) + "\n" + \
        "# parameters\t" + str(self.model.n_params()) + "\n" + \
        "output\t\t\t" + str(self.output_path)
    # TODO "criterion\t\t" + str(self.criterion) + "\n" + \
    # TODO "rnn type\t\t\t" + str(self.model.rnn_type) + "\n" + \
    # TODO "optimizer\t\t" + str(self.optimizer) + "\n" + \
    # TODO "nonlinearity\t" + str(self.model.nonlinearity) + "\n" + \

    def finalize_model(self):
        torch.save(self.model.state_dict(), self.output_path)
        LOGS.write("model saved, training finished\n\n")


    def main(self):
        print("======================================================================")
        print("                              TRAINING")
        print("======================================================================")
        print(self.to_string())
        LOGS.write("=================================================================\n")
        LOGS.write(self.to_string() + "\n")

    def evaluate(self, plot_loss, outputs, batch, mini_prplx, valid_avg_loss):
        avg_loss = plot_loss // EVALUATE_EVERY
        plot_loss = 0
        prplx = self.training_logs(outputs, batch, self.model)
        self.training_prplxs.append(prplx)
        # LOGS.write(str(i) + " batches\n")
        # LOGS.write("train avg loss" + "\t" + str(avg_loss) + "\n")
        # LOGS.write("valid avg loss\t%d\n" % avg_loss)
        # print("train avg loss\t", str(avg_loss))
        # print("valid avg loss\t %d" % valid_avg_loss)
        # print("mini prplx\t\t", mini_prplx)


    def logs(self, n_batches, train_avg_loss, valid_avg_loss, mini_prplx, model_saved):
        # def format_to_length(val, n):
        #     string = str(val)
        #     if len(string) < n:
        #         for i in range(n):
        #             string = "0" + string
        #     return(str)

        LOGS.write("\t" + str(n_batches) + "\t\t")
        LOGS.write("\t\t" + str(train_avg_loss) + "\t")
        LOGS.write("\t" + str(valid_avg_loss) + "\t")
        LOGS.write("\t" + str(mini_prplx) + "\t")
        LOGS.write("\t" + str(model_saved) + "\n")
        print("train avg loss\t", str(train_avg_loss))
        print("valid avg loss\t %d" % valid_avg_loss)
        print("mini prplx\t\t", mini_prplx)


    def test(self):
        total_log_prob = 0
        total_words = 0
        test_epoch = du.CorpusEpoch(self.dm.test, self.dm)
        still_going = True
        while still_going:
            batch, still_going = test_epoch.get_next_batch()
            log_prob, n_words, outputs = self.log_probability(batch)
            total_log_prob += log_prob[0].data[0]
            total_words += n_words
        prplx = ModelUtils.perplexity(total_words, total_log_prob)
        return prplx



    def run_train(self):
        print("======================================================================")
        print("                              TRAINING")
        print("======================================================================")
        print(self.to_string())
        LOGS.write(self.to_string() + "\n")
        LOGS.write("# batches | train avg loss | valid avg loss | valid perplexity | model saved\n" +
                   "----------|----------------|----------------|------------------|-------------\n")
        n_batches = 0
        min_mini_prplx = math.inf
        n_stages_not_converging = 0
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0
            plot_loss = 0
            print("===========================EPOCH %d=============================" % epoch)
            training_lines = du.CorpusEpoch(self.dm.training, self.dm)
            training_still_going = True
            while training_still_going:
                n_batches += 1
                batch, training_still_going = training_lines.get_next_batch()
                outputs, loss = self.train(batch)
                epoch_loss += loss
                plot_loss += loss

                # print epoch number, loss, name, guess
                if n_batches % EVALUATE_EVERY == 0:# and i != 0:
                    print()
                    print(epoch, str(100 * n_batches * training_lines.batch_size // training_lines.n_lines) + "%", time_since(START_TIME))
                    self.training_logs(outputs, batch, self.model)
                    mini_prplx, valid_avg_loss = self.mini_validate()
                    train_avg_loss = plot_loss // EVALUATE_EVERY
                    plot_loss = 0
                    if mini_prplx < min_mini_prplx:
                        n_stages_not_converging = 0
                        self.logs(n_batches, train_avg_loss, valid_avg_loss, mini_prplx, True)
                        # self.model.state_dict()
                        torch.save(self.model.state_dict(), self.output_path + "_" + str(n_batches))
                        min_mini_prplx = mini_prplx
                        print("MINIMUM PERPLEXITY, MODEL SAVED")
                        # self.valid_prplxs.append(prplx)
                    else:
                        self.logs(n_batches, train_avg_loss, valid_avg_loss, mini_prplx, False)
                        n_stages_not_converging += 1
                        if n_stages_not_converging == STOP_TRAINING_THRESHOLD:
                            training_still_going = False
                            self.finalize_model()
                            LOGS.flush()
                    LOGS.flush()
        test_prplx = self.test()
        LOGS.write("TEST PERPLEXITY = " + str(test_prplx))
        self.finalize_model()