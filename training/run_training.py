import sys
import gflags
import my_flags
import models.rnn_classifier
import models.acceptability_cl
import torch

FLAGS = gflags.FLAGS

if __name__ == '__main__':
    print("HELLO WORLD")
    my_flags.get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    # flag_defaults(FLAGS)

    cl = None

    if FLAGS.model_type == "rnn_classifier_pooling":
        cl = models.rnn_classifier.ClassifierPooling(
            hidden_size=FLAGS.hidden_size,
            embedding_size=FLAGS.embedding_size,
            num_layers=FLAGS.num_layers)
        try:
            cl.load_state_dict(torch.load(FLAGS.ckpt_path + FLAGS.experiment_name))
        except IOError:
            pass
        clt = models.rnn_classifier.RNNTrainer(
            FLAGS,
            model=cl)
        clt.run()

    if FLAGS.model_type == "aj_classifier":
        cl = models.acceptability_cl.Classifier(
            hidden_size=FLAGS.hidden_size,
            encoding_size=FLAGS.encoding_size
        )
        encoder = None
        if FLAGS.encoder_type == "rnn_classifier_pooling":
            encoder = models.rnn_classifier.ClassifierPooling(
                hidden_size=FLAGS.encoding_size,
                embedding_size=FLAGS.embedding_size,
                num_layers=FLAGS.encoder_num_layers)
            encoder.load_state_dict(torch.load(FLAGS.encoder_path))
        clt = models.acceptability_cl.AJTrainer(
            FLAGS,
            model=cl,
            encoder=encoder)
        clt.run()

