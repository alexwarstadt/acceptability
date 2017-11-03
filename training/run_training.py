import sys
import gflags
import my_flags
import models.rnn_classifier

FLAGS = gflags.FLAGS

if __name__ == '__main__':
    my_flags.get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    # flag_defaults(FLAGS)

    cl = None

    if FLAGS.model_type == "rnn_classifier_pooling":
        cl = models.rnn_classifier.ClassifierPooling(hidden_size=FLAGS.hidden_size,
                                                     embedding_size=FLAGS.word_embedding_dim,
                                                     num_layers=FLAGS.num_layers)

    clt = models.rnn_classifier.RNNTrainer('../data/discriminator/',
                                           '../data/bnc-30/embeddings_20000.txt',
                                           '../data/bnc-30/vocab_20000.txt',
                                           embedding_size=FLAGS.word_embedding_dim,
                                           model=cl,
                                           stages_per_epoch=FLAGS.stages_per_epoch,
                                           prints_per_stage=FLAGS.prints.per_stage,
                                           convergence_threshold=FLAGS.convergence_threshold,
                                           max_epochs=100,
                                           gpu=FLAGS.gpu,
                                           learning_rate=FLAGS.learning_rate)
    clt.run()
