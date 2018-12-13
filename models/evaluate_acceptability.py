import models.model_trainer as model_trainer
import utils.classifier_data_utils as cdu
import utils.data_processor as dp
from torch.autograd import Variable
import torch
import models.acceptability_cl
import training.my_flags
import gflags
import sys

class Interacter(model_trainer.ModelTrainer):

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
        self.dm = cdu.DataManagerEval(self.FLAGS.test_path, self.FLAGS.embedding_path,
                                          self.FLAGS.vocab_path, self.FLAGS.embedding_size, self.FLAGS.crop_pad_length)
        super(Interacter).__init__(FLAGS, self.classifier)

    def eval(self):
        epoch = cdu.CorpusEpoch(self.dm.test_pairs, self.dm)
        batch = None
        while epoch.still_going:
            batch, _ = epoch.get_new_batch()
            hidden = self.encoder.init_hidden(batch.batch_size)
            input = torch.Tensor(len(batch.tensor_view), batch.batch_size, self.FLAGS.embedding_size)
            for i, t in enumerate(batch.tensor_view):
                input[i] = t
            _, encoding = self.encoder.forward(Variable(input), hidden)
            output = self.classifier.forward(encoding)


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



FLAGS = gflags.FLAGS
training.my_flags.get_flags()

# Parse command line flags.
FLAGS(sys.argv)

i = Interacter(FLAGS)

i.interact(["few information was provided .",
            "mary jumped the horse perfectly over the last fence .",
            "there are arriving three men at that station .",
            "it loved sandy .",
            "who does phineas know a girl who is working with ? "])