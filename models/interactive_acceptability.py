import models.model_trainer as model_trainer
import utils.classifier_data_utils as cdu
import utils.data_processor as dp
from torch.autograd import Variable
import torch
import models.acceptability_cl
import training.my_flags
import gflags
import sys
from itertools import islice




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
            self.encoder = models.rnn_classifier.ClassifierPooling(
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


FLAGS = gflags.FLAGS
training.my_flags.get_flags()

# Parse command line flags.
FLAGS(sys.argv)

i = Interacter(FLAGS)

# i.interact(["I I."])

# i.interactive_mode()

i.interact_file("../acceptability_corpus/temp")


# FLAGS
# --encoding_size
# 1034
# --embedding_size
# 300
# --data_type
# discriminator
# --vocab_path
# /Users/alexwarstadt/Workspace/data/bnc-30/vocab_20000.txt
# --log_path
# /Users/alexwarstadt/Workspace/acceptability/logs/interactive_output
# --crop_pad_length
# 30
# --embedding_path
# /Users/alexwarstadt/Workspace/data/bnc-30/embeddings_20000.txt
# --encoder_type
# rnn_classifier_pooling
# --encoder_num_layers
# 3
# --classifier_type
# aj_classifier
# --hidden_size
# 21
# --learning_rate
# .0001
# --encoder_path
# /Users/alexwarstadt/Workspace/acceptability/checkpoints/CPU_sweep_1106235815_rnn_classifier_pooling_16-lr8e-05-h_size1034-datadiscriminator-num_layers3
# --classifier_path
# /Users/alexwarstadt/Workspace/acceptability/checkpoints/CPU_sweep_1229132558_aj_classifier_57-lr8.8e-05-h_size21-dataaj_balanced-encCPU_sweep_1106235815_rnn_classifier_pooling_16-lr8e-05-h_size1034-datadiscriminator-num_layers3