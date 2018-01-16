import models.rnn_classifier
import torch
import os

# A script for converting encoders of type LSTMPoolingClassifier from GPU to CPU


dirs = ["sweep_1229145144",
        "sweep_1227010500",
        "sweep_1229132558"]


for dir in dirs:
    for model in os.listdir("/scratch/asw462/models/" + dir + "/"):

        h_size = int(filter(lambda s: s.startswith("h_size"), model.split("-"))[0].replace("h_size", ""))
        num_layers = int(filter(lambda s: s.startswith("num_layers"), model.split("-"))[0].replace("num_layers", ""))

        encoder = models.rnn_classifier.LSTMPoolingClassifier(
            hidden_size=h_size,
            embedding_size=300,
            num_layers=num_layers)

        print '/scratch/asw462/models/' + dir + "/" + model

        try:
            encoder.load_state_dict(torch.load(
                '/scratch/asw462/models/' + dir + "/" + model))
            encoder.cpu()
            torch.save(encoder.state_dict(), '/scratch/asw462/models/' + dir + '/CPU_' + model)
        except KeyError:
            pass
