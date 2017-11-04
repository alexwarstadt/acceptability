import sys
import gflags
import my_flags
import models.rnn_classifier

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

    clt = models.rnn_classifier.RNNTrainer(
        FLAGS,
        model=cl)
    clt.run()
