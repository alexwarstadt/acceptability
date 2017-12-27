import gflags

def get_flags():
    # Debug settings.
    gflags.DEFINE_string("data_dir",
                         "/scratch/asw462/data/bnc_lm/",
                         "dir containing train.txt, test.txt, valid.txt")
    gflags.DEFINE_string("vocab_path",
                         "/scratch/asw462/data/bnc-30/vocab_20000.txt",
                         "vocab text file")
    gflags.DEFINE_string("embedding_path",
                         "/scratch/asw462/data/bnc-30/embeddings_20000.txt",
                         "embeddings file, must match vocab")
    gflags.DEFINE_string("log_path", "/scratch/asw462/logs/", "")
    gflags.DEFINE_string("data_type", "discriminator", "figure out how to use this")
    gflags.DEFINE_string("model_type",
                         "rnn_classifier_pooling",
                         "options: rnn_classifier_pooling, acceptability_classifier,....")
    gflags.DEFINE_string("ckpt_path", "/scratch/asw462/models/", "")
    gflags.DEFINE_boolean("gpu", False, "set to false on local")
    gflags.DEFINE_string("experiment_name", "", "")

    #sizes
    gflags.DEFINE_integer("embedding_size", 300, "")
    gflags.DEFINE_integer("crop_pad_length", 30, "")

    #chunks
    gflags.DEFINE_integer("stages_per_epoch",
                          100,
                          "how many eval/stats steps per epoch?")
    gflags.DEFINE_integer("prints_per_stage",
                          1,
                          "how often to print stats to stdout during epoch")
    gflags.DEFINE_integer("convergence_threshold",
                          20,
                          "how many eval steps before early stop")
    gflags.DEFINE_integer("max_epochs",
                          10,
                          "number of epochs before stop, essentially unreachable")
    gflags.DEFINE_integer("batch_size", 32, "")
    gflags.DEFINE_boolean("by_source", False, "will output stats be broken down by source")

    #tunable parameters
    gflags.DEFINE_integer("hidden_size", 300, "")
    gflags.DEFINE_integer("num_layers", 1, "")
    gflags.DEFINE_float("learning_rate", .0005, "")

    #aj flags
    gflags.DEFINE_integer("encoding_size", 100, "output size of encoder, input size of aj")
    gflags.DEFINE_integer("encoder_num_layers", 1, "number of layers in the encoder network")
    gflags.DEFINE_string("encoder_path", "", "location of encoder checkpoint")
    gflags.DEFINE_string("encoder_type", "rnn_classifier_pooling", "the class of the encoder model")

