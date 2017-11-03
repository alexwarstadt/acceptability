import random
import time

import utils.data_utils as du
import torch.nn as nn
from utils.classifier_utils import *
from utils.process_corpus import crop_sentences
from torch.autograd import Variable

import lm as my_lm
import rnn_classifier
from utils import classifier_data_utils as cdu

EVALUATE_EVERY = 1000
LOGS = open("logs/rnn-acceptability-logs", "a")
OUTPUT_PATH = "models/rnn-acceptability_classifier"


class Classifier(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size):
        super(Classifier, self).__init__()
        self.hidden_size = hidden_size
        # self.lm2h = nn.Linear(lm.hidden_size, self.hidden_size)
        self.e2h = nn.LSTM(encoder_hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()


    def forward(self, input, hidden_states):
        o, hc = self.e2h(input, hidden_states)
        output = self.sigmoid(self.h2o(o[-1]))
        return output, o

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.hidden_size)),
                Variable(torch.zeros(1, batch_size, self.hidden_size)))




class ClTrainer():
    def __init__(self, corpus_path, embedding_path, embedding_size, model, encoder, encoder_path, learning_rate=.005):
        self.corpus_path = corpus_path
        self.model = model
        self.encoder = encoder
        self.embedding_size = embedding_size
        self.encoder_path = encoder_path
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # or use RMSProp
        self.dm = cdu.DataManager(corpus_path, embedding_path, embedding_size)
        now = time.localtime()
        time_stamp = str(now.tm_mon) + "-" + str(now.tm_mday) + "_" \
                     + str(now.tm_hour) + ":" + str(now.tm_min) + ":" + str(now.tm_sec)
        self.output_path = OUTPUT_PATH + "_" + time_stamp

    def to_string(self):
        return "data\t\t\t" + self.corpus_path + "\n" + \
            "embedding size\t" + str(self.embedding_size) + "\n" + \
            "encoder size\t" + str(self.encoder.hidden_size) + "\n" + \
            "hidden size\t\t" + str(self.model.hidden_size) + "\n" + \
            "learning rate\t" + str(self.learning_rate) + "\n" + \
            "output\t\t\t" + str(self.output_path)

    def encode(self, batch):
        hidden = self.encoder.init_hidden(batch.batch_size)
        input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.embedding_size)
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        _, encodeds = self.encoder.forward(Variable(input), hidden)
        return encodeds

    def encode_lm(self, batch):
        hidden = self.encoder.init_hidden(batch.batch_size)
        #TODO
        input = batch.tensor_view
        hidden = self.encoder.init_hidden(batch.batch_size)
        hiddens = [hidden[-1][0]]
        for word_batch in input[:-1]:
            output, hidden = self.encoder.forward(Variable(word_batch), hidden)
            hiddens.append(hidden[-1][0].data)
        return hiddens

    def train(self, batch, targets):
        inputs = self.encode(batch)
        hidden = self.model.init_hidden(batch.batch_size)
        outputs, _ = self.model.forward(inputs, hidden)
        loss = self.get_batch_loss(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return outputs, loss.data[0]

    # def get_outputs(self, batch):
    #     sentence_vecs = self.get_sentence_vecs(batch)


    def get_batch_loss(self, outputs, output_targets):
        ce_loss = torch.nn.BCELoss()
        ce_loss.weight = torch.Tensor([self.dm.corpus_bias for i in range(len(output_targets))])
        loss = Variable(torch.Tensor([0]))
        # for out, target in zip(outputs, output_targets):
            # loss += self.my_CELoss(out, target)
        loss = ce_loss(outputs, Variable(torch.FloatTensor(output_targets)))
        return loss

    def batch_accuracy(self, outputs, output_targets):
        true_poz = 0
        false_poz = 0
        true_neg = 0
        false_neg = 0
        for out, target in zip(outputs, output_targets):
            if out.data[0] < .5:
                if target < .5:
                    true_neg += 1
                else:
                    false_neg += 1
            elif out.data[0] >= .5:
                if target >= .5:
                    true_poz += 1
                else:
                    false_poz += 1
        return true_poz, false_poz, true_neg, false_neg


    def validate(self):
        valid_epoch = cdu.CorpusEpoch(self.dm.valid, self.dm)
        random.shuffle(self.dm.valid)
        still_going = True
        total_loss = 0
        n_batches = 0
        true_poz, false_poz, true_neg, false_neg = 0, 0, 0, 0
        while still_going:
            n_batches += 1
            batch, targets, still_going = valid_epoch.get_next_batch()
            inputs = self.encode(batch)
            hidden = self.model.init_hidden(batch.batch_size)
            outputs, _ = self.model.forward(inputs, hidden)
            loss = self.get_batch_loss(outputs, targets)
            total_loss += loss
            tp, fp, tn, fn = self.batch_accuracy(outputs, targets)
            true_poz += tp
            false_poz += fp
            true_neg += tn
            false_neg += fn
        v_f1 = f1(true_poz, false_poz, true_neg, false_neg)
        v_matthews = matthews(true_poz, false_poz, true_neg, false_neg)
        print("valid stats\t", "tp =", true_poz/len(valid_epoch.data_pairs), "fp =", false_poz/len(valid_epoch.data_pairs),
              "tn =", true_neg/len(valid_epoch.data_pairs), "fn =", false_neg/len(valid_epoch.data_pairs))
        print("f1\t\t\t\t", v_f1)
        print("matthews\t\t", v_matthews)
        print_min_and_max(outputs, batch)
        return total_loss.data[0] / len(valid_epoch.data_pairs), v_f1, v_matthews

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


    # def f1(self, tp, fp, tn, fn):
    #     return 2*tp / ((2*tp) + fp + fn)
    #
    # def matthews(self, tp, fp, tn, fn):
    #     return ((tp*tn) - (fp*fn)) / math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+tn))

    def interact(self, sentences):
        cropped_sentences = crop_sentences(sentences)
        batch = cdu.Batch(cropped_sentences, self.dm)
        sentence_vecs = self.get_sentence_vecs_without_stop_lstm(batch)
        outputs = self.model.forward(sentence_vecs, self.encoder)
        for s, o in zip(sentences, outputs):
            print(o[0].data, s)

    def test(self):
        test_epoch = cdu.CorpusEpoch(self.dm.test, self.dm)
        random.shuffle(self.dm.test)
        output_log = open("logs/output_" + self.output_path.split("/")[1], "w")
        still_going = True
        total_loss = 0
        n_batches = 0
        true_poz, false_poz, true_neg, false_neg = 0, 0, 0, 0
        while still_going:
            n_batches += 1
            batch, targets, still_going = test_epoch.get_next_batch()
            outputs, loss = self.train(batch, targets)
            self.log_output(outputs, targets, batch, output_log)
            loss = self.get_batch_loss(outputs, targets)
            total_loss += loss
            tp, fp, tn, fn = self.batch_accuracy(outputs, targets)
            true_poz += tp
            false_poz += fp
            true_neg += tn
            false_neg += fn
        t_f1 = f1(true_poz, false_poz, true_neg, false_neg)
        t_matthews = matthews(true_poz, false_poz, true_neg, false_neg)
        LOGS.write("-------TEST-------\n")
        LOGS.write("test confusion\t" + "tp=" + "{0:.4g}".format(true_poz / len(test_epoch.data_pairs)) +
                   " fp=" + "{0:.4g}".format(false_poz / len(test_epoch.data_pairs)) +
                   " tn=" + "{0:.4g}".format(true_neg / len(test_epoch.data_pairs)) +
                   " fn=" + "{0:.4g}".format(false_neg / len(test_epoch.data_pairs)) + "\n")
        LOGS.write("test f1 =" + " " + "{0:.4g}".format(t_f1) + "\n")
        LOGS.write("test matthews =" + " " + "{0:.4g}".format(t_matthews) + "\n")
        print("==========TEST==========\n")
        print("test stats\t", "tp =", true_poz / len(test_epoch.data_pairs), "fp =",
              false_poz / len(test_epoch.data_pairs),
              "tn =", true_neg / len(test_epoch.data_pairs), "fn =", false_neg / len(test_epoch.data_pairs))
        print("f1\t\t\t\t", t_f1)
        print("matthews\t\t", t_matthews)
        print_min_and_max(outputs, batch)
        return total_loss.data[0] / len(test_epoch.data_pairs), t_f1, t_matthews


    def log_output(self, outputs, targets, batch, log):
        for o, t, s in zip(outputs, targets, batch.sentences_view):
            log.write(str(o.data[0]) + "\t" + str(t) + "\t" + s + "\n")


    def run_train(self):
        print("======================================================================")
        print("                              TRAINING")
        print("======================================================================")
        print(self.to_string())
        LOGS.write("\n\n" + self.to_string() + "\n")
        LOGS.write("# epoch | train avg loss | valid avg loss | t matthews | v matthews | t f1 | v f1 | model saved\n" +
                   "--------|----------------|----------------|------------|------------|------|------|------------\n")
        n_batches = 0
        n_epochs = 100
        plot_loss = 0
        n_epochs_not_converging = 0
        max_matthews = 0
        epoch = 0
        benchmark = ""
        while epoch < n_epochs and n_epochs_not_converging < 20:
            epoch += 1
            epoch_loss = 0
            tp, fp, tn, fn = 0, 0, 0, 0
            # plot_loss = 0
            print("===========================EPOCH %d=============================" % epoch)
            training_epoch = cdu.CorpusEpoch(self.dm.training, self.dm)
            random.shuffle(self.dm.training)
            epoch_still_going = True
            while epoch_still_going:
                n_batches += 1
                batch, targets, epoch_still_going = training_epoch.get_next_batch()
                outputs, loss = self.train(batch, targets)
                plot_loss += loss
                epoch_loss += loss
                _tp, _fp, _tn, _fn = self.batch_accuracy(outputs, targets)
                tp += _tp
                fp += _fp
                tn += _tn
                fn += _fn

                # if n_batches % EVALUATE_EVERY == 0:  # and i != 0:
                #     print(plot_loss)
                #     print([x.data[0] for x in outputs])
                #     self.print_min_and_max(outputs, batch)
                #     plot_loss = 0
            t_f1 = f1(tp, fp, tn, fn)
            t_matthews = matthews(tp, fp, tn, fn)
            epoch_avg_loss = epoch_loss / len(self.dm.training)
            print("epoch avg loss\t\t", epoch_avg_loss)
            print("training stats\t", "tp =", tp/len(self.dm.training), "fp =", fp/len(self.dm.training),
              "tn =", tn/len(self.dm.training), "fn =", fn/len(self.dm.training))
            print("f1\t\t\t\t", t_f1)
            print("matthews\t\t", t_matthews)
            print("-------valid-------")
            valid_avg_loss, v_f1, v_matthews = self.validate()
            print("valid avg loss\t\t", valid_avg_loss)
            if v_matthews > max_matthews:
                n_epochs_not_converging = 0
                torch.save(self.model.state_dict(), self.output_path + "_" + str(n_batches))
                benchmark = self.output_path + "_" + str(n_batches)
                max_matthews = v_matthews
                print("MAX MATTHEWS, MODEL SAVED")
                logs(LOGS, epoch, epoch_avg_loss, valid_avg_loss, t_matthews, v_matthews, t_f1, v_f1, True)
            else:
                n_epochs_not_converging += 1
                logs(LOGS, epoch, epoch_avg_loss, valid_avg_loss, t_matthews, v_matthews, t_f1, v_f1, False)
            LOGS.flush()
        try:
            self.model.load_state_dict(torch.load(benchmark))
        except FileNotFoundError:
            pass
        _, test_f1, test_matthews = self.test()





#============= LOAD LM FROM MODEL =============
# encoder = model.MyLSTM(input_size=300, hidden_size=350, output_size=20001, n_layers=1, nonlinearity=nn.Tanh())
# encoder.load_state_dict(torch.load('models/model_7-14_15:33:4_110500'))

encoder_path = 'models/rnn_classifier_7-28_17:50:13_85975'
encoder = rnn_classifier.Classifier(hidden_size=350, embedding_size=300)
encoder.load_state_dict(torch.load(encoder_path))



#============= EXPERIMENT ================
size_range = (10, 80)
lr = (1, 4)
corpus = ['acceptability_corpus/corpus_table_tokenized_crop30', 'acceptability_corpus/just_levin']
# corpus = ['acceptability_corpus/corpus_table_tokenized_crop30']

for _ in range(50):
    cl = Classifier(random.randint(size_range[0], size_range[1]), encoder.hidden_size)
    clt = ClTrainer(corpus[random.randint(0, len(corpus)-1)],
                    'embeddings/glove.6B.300d.txt', 300, cl,
                    encoder, encoder_path, learning_rate=math.pow(.2, random.uniform(lr[0], lr[1])))
    clt.run_train()

#     dim_hidden = random.randint(hidden_size_range[0], hidden_size_range[1])





#============= TRAIN CLASSIFIER =============
# cl = Classifier(20, encoder.hidden_size)
# clt = ClTrainer('acceptability_corpus/corpus_table_tokenized_crop30', 'embeddings/glove.6B.300d.txt', 300, cl, encoder, encoder_path, learning_rate=.015)
# clt.run_train()




#============= GENERATE =============
dm = du.DataManager('../data/bnc/bnc.txt', 'embeddings/glove.6B.300d.txt', 300, crop_pad_length=30)
mu = my_lm.ModelUtils(dm)
out = open("acceptability_corpus/lm_generated2", "a")

for _ in range(131):
    lines = ""
    for __ in range(1000):
        lines += ("lm	0	*	<s> " + mu.generate_sans_probability(29, encoder) + "</s>\n")
    out.write(lines)
out.close()



# clt.model.load_state_dict(torch.load("models/acceptability_classifier_7-24_1:22:7_8415"))
# clt.interact([
#     "john likes mary .",
#     "likes john mary .",
#     "the sky is very blue today .",
#     "very blue today the sky is .",
#     "john has a fondness for jane .",
#     "john has a fondness with jane .",
#     "john has a fondness about jane .",
#     "every sand is hot .",
#     "the water trickled down the road ."
# ])



#============= FIND LABEL BIAS =============