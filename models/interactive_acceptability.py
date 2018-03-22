import models.model_trainer as model_trainer
import utils.classifier_data_utils as cdu
import utils.data_processor as dp
from torch.autograd import Variable
import torch
import models.acceptability_cl
import training.my_flags
import gflags
import sys
import math
from itertools import islice




def matthews(tp, tn, fp, fn):
    """tp*tn - fp*fn / sqrt( tp+fp tp+fn tn+fp tn+fn )"""
    try:
        m = float((tp * tn) - (fp * fn)) / \
            math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    except ZeroDivisionError:
        m = 0
    return m


class Interacter():

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.classifier = None
        if FLAGS.classifier_type == "aj_classifier":
            self.classifier = models.acceptability_cl.AcceptabilityClassifier(
                hidden_size=FLAGS.hidden_size,
                encoding_size=FLAGS.encoding_size
            )
        self.encoder = None
        if FLAGS.encoder_type == "rnn_classifier_pooling":
            self.encoder = models.rnn_classifier.LSTMPoolingClassifier(
                hidden_size=FLAGS.encoding_size,
                embedding_size=FLAGS.embedding_size,
                num_layers=FLAGS.encoder_num_layers)
        self.encoder.load_state_dict(torch.load(FLAGS.encoder_path))
        self.classifier.load_state_dict(torch.load(FLAGS.classifier_path))
        self.dm = cdu.InteractiveDataManager(self.FLAGS.data_dir, self.FLAGS.embedding_path,
                                          self.FLAGS.vocab_path, self.FLAGS.embedding_size, self.FLAGS.crop_pad_length)

    def interact(self, sentences):
        sanitized_sentences = [dp.sanitize_sentence(s.strip()) for s in sentences]
        data_pairs = [(s, -1, -1) for s in sanitized_sentences]
        batch = cdu.Batch(data_pairs, self.dm)
        hidden = self.encoder.init_hidden(batch.batch_size)
        input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.FLAGS.embedding_size)
        for i, t in enumerate(batch.tensor_view):
            input[i] = t
        _, encoding = self.encoder.forward(Variable(input), hidden)
        output = self.classifier.forward(encoding)
        for s, o in zip(sentences, output):
            print ("%.2f" % o[0].data[0]) + "\t" + s.strip()
        return [o[0].data[0] for o in output]

    def interact_one(self, sentence):
        self.interact([sentence])

    def interactive_mode(self):
        while True:
            s = raw_input("\nYour sentence: ")
            self.interact_one(s)

    def interact_file(self, path):
        with open(path) as f:
            while True:
                next_n_lines = list(islice(f, 32))
                if not next_n_lines:
                    break
                self.interact(next_n_lines)

    def evaluate_file(self, path):
        gold_labels = []
        guesses = []
        with open(path) as f:
            while True:
                next_n_lines = list(islice(f, 32))
                gold_labels.extend([float(line.split("\t")[1]) for line in next_n_lines])
                if not next_n_lines:
                    break
                guesses.extend(self.interact([line.split("\t")[3] for line in next_n_lines]))
        (tp, tn, fp, fn) = (0, 0, 0, 0)
        for guess, gold in zip(guesses, gold_labels):
            if gold == 1 and guess >= 0.5:
                tp += 1
            if gold == 0 and guess < 0.5:
                tn += 1
            if gold == 1 and guess < 0.5:
                fn += 1
            if gold == 0 and guess >= 0.5:
                fp += 1
        test_matt = matthews(tp, tn, fp, fn)
        print("Matthews: " + str(test_matt))
        print("Accuracy: " + str((float(tp+tn)/float(tp+tn+fp+fn))))
        print("tp=%d, tn=%d, fp=%d, fn=%d" % (tp, tn, fp, fn))



FLAGS = gflags.FLAGS
training.my_flags.get_flags()

# Parse command line flags.
FLAGS(sys.argv)

i = Interacter(FLAGS)

i.evaluate_file("../acceptability_corpus/artificial/svo.tsv")


# i.interact_file("../acceptability_corpus/test_tokenized.tsv")


