import gflags

def get_flags():
    # Debug settings.
    gflags.DEFINE_string("data_dir",
                         "/scratch/asw462/data/bnc_lm/",
                         "dir containing train.txt, test.txt, valid.txt")
    gflags.DEFINE_string("vocab_path",
                         "/scratch/asw462/data/bnc-30/vocab_20000.txt",
                         "vocab text file")
    gflags.DEFINE_string("embedding_data_path",
                         "/scratch/asw462/data/bnc-30/embeddings_20000.txt",
                         "embeddings file, must match vocab")
    gflags.DEFINE_string("log_path", "/scratch/asw462/logs/", "")
    gflags.DEFINE_string("data_type", "discriminator", "figure out how to use this")
    gflags.DEFINE_string("model_type",
                         "rnn_classifier_pooling",
                         "options: rnn_classifier_pooling, acceptability_classifier,....")
    gflags.DEFINE_string("ckpt_path", "/scratch/asw462/models/", "")
    gflags.DEFINE_bool("gpu", True, "set to false on local")

    #sizes
    gflags.DEFINE_integer("word_embedding_dim", 300, "")
    gflags.DEFINE_integer("crop_pad_length", 30, "")

    #chunks
    gflags.DEFINE_integer("stages_per_epoch",
                          "100",
                          "how many eval/stats steps per epoch?")
    gflags.DEFINE_integer("prints_per_stage",
                          "1",
                          "how often to print stats to stdout during epoch")
    gflags.DEFINE_integer("convergence_threshold",
                          "20",
                          "how many eval steps before early stop")
    gflags.DEFINE_integer("batch_size", "32", "")

    #tunable parameters
    gflags.DEFINE_integer("hidden_size", None, "")
    gflags.DEFINE_integer("num_layers", 1, "")
    gflags.DEFINE_float("learning_rate", .0005, "")









    gflags.DEFINE_bool(
        "debug",
        False,
        "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool(
        "show_progress_bar",
        True,
        "Turn this off when running experiments on HPC.")
    gflags.DEFINE_string("git_branch_name", "", "Set automatically.")
    gflags.DEFINE_string("slurm_job_id", "", "Set automatically.")
    gflags.DEFINE_integer(
        "deque_length",
        100,
        "Max trailing examples to use when computing average training statistics.")
    gflags.DEFINE_string("git_sha", "", "Set automatically.")
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("load_experiment_name", None, "")

    # Data types.
    gflags.DEFINE_enum("data_type",
                       "bl",
                       ["bl",
                        "sst",
                        "sst-binary",
                        "nli",
                        "arithmetic",
                        "listops",
                        "sign",
                        "eq",
                        "relational"],
                       "Which data handler and classifier to use.")

    # Choose Genre.
    # 'fiction', 'government', 'slate', 'telephone', 'travel'
    # 'facetoface', 'letters', 'nineeleven', 'oup', 'verbatim'
    gflags.DEFINE_string("train_genre", None, "Filter MultiNLI data by genre.")
    gflags.DEFINE_string("eval_genre", None, "Filter MultiNLI data by genre.")

    # Where to store checkpoints
    gflags.DEFINE_string(
        "log_path",
        "./logs",
        "A directory in which to write logs.")
    gflags.DEFINE_string(
        "load_log_path",
        None,
        "A directory in which to write logs.")
    gflags.DEFINE_boolean(
        "write_proto_to_log",
        False,
        "Write logs in a protocol buffer format.")
    gflags.DEFINE_string(
        "ckpt_path", None, "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string(
        "metrics_path",
        None,
        "A directory in which to write metrics.")
    gflags.DEFINE_integer(
        "ckpt_step",
        1000,
        "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_boolean(
        "load_best",
        False,
        "If True, attempt to load 'best' checkpoint.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string(
        "eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("seq_length", 200, "")
    gflags.DEFINE_boolean(
        "allow_cropping",
        False,
        "Trim overly long training examples to fit. If not set, skip them.")
    gflags.DEFINE_integer("eval_seq_length", None, "")
    gflags.DEFINE_boolean(
        "allow_eval_cropping",
        False,
        "Trim overly long evaluation examples to fit. If not set, crash on overly long examples.")
    gflags.DEFINE_boolean(
        "smart_batching",
        True,
        "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer(
        "eval_data_limit",
        None,
        "Truncate evaluation set to this many batches. -1 indicates no truncation.")
    gflags.DEFINE_boolean(
        "bucket_eval",
        True,
        "Bucket evaluation data for speed improvement.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer(
        "shuffle_eval_seed",
        123,
        "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
                         "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum(
        "model_type", "RNN", [
            "CBOW", "RNN", "SPINN", "RLSPINN", "ChoiPyramid"], "")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_boolean("lowercase", False, "When True, ignore case.")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse.")
    gflags.DEFINE_boolean(
        "validate_transitions",
        True,
        "Constrain predicted transitions to ones that give a valid parse tree.")
    gflags.DEFINE_float(
        "embedding_keep_rate",
        0.9,
        "Used for dropout on transformed embeddings and in the encoder RNN.")
    gflags.DEFINE_boolean("use_l2_loss", True, "")
    gflags.DEFINE_boolean("use_difference_feature", True, "")
    gflags.DEFINE_boolean("use_product_feature", True, "")

    # Tracker settings.
    gflags.DEFINE_integer(
        "tracking_lstm_hidden_dim",
        None,
        "Set to none to avoid using tracker.")
    gflags.DEFINE_boolean(
        "tracking_ln",
        False,
        "When True, layer normalization is used in tracking.")
    gflags.DEFINE_float(
        "transition_weight",
        None,
        "Set to none to avoid predicting transitions.")
    gflags.DEFINE_boolean("lateral_tracking", True,
                          "Use previous tracker state as input for new state.")
    gflags.DEFINE_boolean(
        "use_tracking_in_composition",
        True,
        "Use tracking lstm output as input for the reduce function.")
    gflags.DEFINE_boolean(
        "composition_ln",
        True,
        "When True, layer normalization is used in TreeLSTM composition.")
    gflags.DEFINE_boolean("predict_use_cell", True,
                          "Use cell output as feature for transition net.")

    # Reduce settings.
    gflags.DEFINE_enum(
        "reduce", "treelstm", [
            "treelstm", "treegru", "tanh"], "Specify composition function.")

    # Pyramid model settings
    gflags.DEFINE_boolean(
        "pyramid_trainable_temperature",
        None,
        "If set, add a scalar trained temperature parameter.")
    gflags.DEFINE_float("pyramid_temperature_decay_per_10k_steps",
                        0.5, "What it says on the box.")
    gflags.DEFINE_float(
        "pyramid_temperature_cycle_length",
        0.0,
        "For wake-sleep-style experiments. 0.0 disables this feature.")

    # Encode settings.
    gflags.DEFINE_enum("encode",
                       "projection",
                       ["pass",
                        "projection",
                        "gru",
                        "attn"],
                       "Encode embeddings with sequential context.")
    gflags.DEFINE_boolean("encode_reverse", False, "Encode in reverse order.")
    gflags.DEFINE_boolean(
        "encode_bidirectional",
        False,
        "Encode in both directions.")
    gflags.DEFINE_integer(
        "encode_num_layers",
        1,
        "RNN layers in encoding net.")

    # RL settings.
    gflags.DEFINE_float(
        "rl_mu",
        0.1,
        "Use in exponential moving average baseline.")
    gflags.DEFINE_enum("rl_baseline",
                       "ema",
                       ["ema",
                        "pass",
                        "greedy",
                        "value"],
                       "Different configurations to approximate reward function.")
    gflags.DEFINE_enum("rl_reward", "standard", ["standard", "xent"],
                       "Different reward functions to use.")
    gflags.DEFINE_float("rl_weight", 1.0, "Hyperparam for REINFORCE loss.")
    gflags.DEFINE_boolean("rl_whiten", False, "Reduce variance in advantage.")
    gflags.DEFINE_boolean(
        "rl_valid",
        True,
        "Only consider non-validated actions.")
    gflags.DEFINE_float(
        "rl_epsilon",
        1.0,
        "Percent of sampled actions during train time.")
    gflags.DEFINE_float(
        "rl_epsilon_decay",
        50000,
        "Step constant in epsilon delay equation.")
    gflags.DEFINE_float(
        "rl_confidence_interval",
        1000,
        "Penalize probabilities of transitions.")
    gflags.DEFINE_float(
        "rl_confidence_penalty",
        None,
        "Penalize probabilities of transitions.")
    gflags.DEFINE_boolean(
        "rl_catalan",
        False,
        "Sample over a uniform distribution of binary trees.")
    gflags.DEFINE_boolean(
        "rl_catalan_backprop",
        False,
        "Sample over a uniform distribution of binary trees.")
    gflags.DEFINE_boolean(
        "rl_wake_sleep",
        False,
        "Inverse relationship between temperature and rl_weight.")
    gflags.DEFINE_boolean(
        "rl_transition_acc_as_reward",
        False,
        "Use the transition accuracy as the reward. For debugging only.")

    # MLP settings.
    gflags.DEFINE_integer(
        "mlp_dim",
        1024,
        "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_mlp_layers", 2, "Number of MLP layers.")
    gflags.DEFINE_boolean(
        "mlp_ln",
        True,
        "When True, layer normalization is used between MLP layers.")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
                        "Used for dropout in the semantic task classifier.")

    # Optimization settings.
    gflags.DEFINE_enum(
        "optimizer_type", "Adam", [
            "Adam", "RMSprop", "YellowFin"], "")
    gflags.DEFINE_integer(
        "training_steps",
        500000,
        "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_float(
        "learning_rate_decay_per_10k_steps",
        0.75,
        "Used in optimizer.")
    gflags.DEFINE_boolean(
        "actively_decay_learning_rate",
        True,
        "Used in optimizer.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float(
        "init_range",
        0.005,
        "Mainly used for softmax parameters. Range for uniform random init.")

    # Display settings.
    gflags.DEFINE_integer(
        "statistics_interval_steps",
        100,
        "Log training set performance statistics at this interval.")
    gflags.DEFINE_integer(
        "eval_interval_steps",
        100,
        "Evaluate at this interval.")
    gflags.DEFINE_integer(
        "sample_interval_steps",
        None,
        "Sample transitions at this interval.")
    gflags.DEFINE_integer("ckpt_interval_steps", 5000,
                          "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean(
        "ckpt_on_best_dev_error",
        True,
        "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")
    gflags.DEFINE_integer(
        "early_stopping_steps_to_wait",
        25000,
        "If development set error doesn't improve significantly in this many steps, cease training.")
    gflags.DEFINE_boolean("evalb", False, "Print transition statistics.")
    gflags.DEFINE_integer("num_samples", 0, "Print sampled transitions.")

    # Evaluation settings
    gflags.DEFINE_boolean(
        "expanded_eval_only_mode",
        False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")  # TODO: Rename.
    gflags.DEFINE_boolean(
        "expanded_eval_only_mode_use_best_checkpoint",
        True,
        "When in expanded_eval_only_mode, load the ckpt_best checkpoint.")
    gflags.DEFINE_boolean("write_eval_report", False, "")
    gflags.DEFINE_boolean(
        "eval_report_use_preds", True, "If False, use the given transitions in the report, "
        "otherwise use predicted transitions. Note that when predicting transitions but not using them, the "
        "reported predictions will look very odd / not valid.")  # TODO: Remove.

    # Evolution Strategy
    gflags.DEFINE_boolean(
        "transition_detach",
        False,
        "Detach transition decision from backprop.")
    gflags.DEFINE_boolean("evolution", False, "Use evolution to train parser.")
    gflags.DEFINE_float(
        "es_sigma",
        0.05,
        "Standard deviation for Gaussian noise.")
    gflags.DEFINE_integer(
        "es_num_episodes",
        2,
        "Number of simultaneous episodes to run.")
    gflags.DEFINE_integer(
        "es_num_roots",
        2,
        "Number of simultaneous episodes to run.")
    gflags.DEFINE_integer("es_episode_length", 1000, "Length of each episode.")
    gflags.DEFINE_integer("es_steps", 1000, "Number of evolution steps.")
    gflags.DEFINE_boolean(
        "mirror",
        False,
        "Do mirrored/antithetic sampling. If doing mirrored sampling, number of perturbtations will be double es_num_episodes.")
    gflags.DEFINE_float(
        "eval_sample_size",
        None,
        "Percentage (eg 0.3) of batches to be sampled for evaluation during training (only for ES). If None, use all.")