from __future__ import print_function
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Acceptability Judgments")
    parser.add_argument("-d", "--data_dir", type=str, default="./data",
                        help="Directory containing train.txt, test.txt" +
                        "and valid.txt")
    parser.add_argument("-e", "--embedding", type=str, default="Glove.300d",
                        help="Embedding type to be used")
    parser.add_argument("-l", "--logs", type=str, default="./logs",
                        help="Log directory")
    parser.add_argument("-dt", "--data_type", type=str,
                        default="discriminator",
                        help="Data type")
    # TODO: Take a look on how to make this enum later
    parser.add_argument("-m", "--model", type=str,
                        default="lstm_pooling_classifier",
                        help="Type of the model you want to use")
    parser.add_argument("-s", "--save_loc", type=str, default="./models",
                        help="Save point for models")
    parser.add_argument("-g", "--gpu", type=bool, default=False,
                        "Whether use GPU or not")
    parser.add_argument("-p", "--crop_pad_length", type=int, default=30,
                        help="Padding Crop length")

    # Chunk parameters
    parser.add_argument("--stages_per_epoch", type=int, default=100,
                        help="Eval/Stats steps per epoch")
    parser.add_argument("--prints_per_stage", type=int, default=1,
                        help="How many times print stats per epoch")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--by_source", type=bool, default=False,
                        help="Break output stats by source")
    parser.add_argument("--max_pool", type=bool, default=False,
                        help="Use max pooling for CBOW")

    # Tuneable parameters
    parser.add_argument("-h", "--hidden_size", type=int, default=300,
                        help="Hidden dimension for LSTM")
    parser.add_argument("--num_layers", type=int, default=1,
                        help="Number of layers for LSTM")
    parser.add_argument("-lr", "--learning_rate", type=float, default=.0005,
                        help="Learning rate")

    parser.add_argument("--encoding_size", type=int, default=100,
                        help="Output size of encoder, input size of aj")
    parser.add_argument("--encoding_num_layers", type=int, default=1,
                        help="Number of layers in encoder network")
    parser.add_argument("--encoder_path", type=str, default="",
                        help="Location of encoder checkpoint")
    parser.add_argument("--encoding_type", type=str,
                        default="lstm_pooling_classifier",
                        help="Class of encoder")

    parser.add_argument("--classifier_path", type=str, default="",
                        help="Location of classifier checkpoint")
    parser.add_argument("--classifier_type", type=str, default="aj_classifier",
                        help="Class of classifier model")

    parser.add_argument("--test_path", type=str, default="",
                        help="Location of test set")
    return parser
