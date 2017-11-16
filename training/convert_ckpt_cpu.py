import models.rnn_classifier
import torch
import os

for model in os.listdir("/scratch/asw462/models/"):
    h_size = int(filter(lambda s: s.startswith("h_size"), model.split("-"))[0].replace("h_size", ""))
    num_layers = int(filter(lambda s: s.startswith("num_layers"), model.split("-"))[0].replace("num_layers", ""))

    encoder = models.rnn_classifier.ClassifierPooling(
        hidden_size=h_size,
        embedding_size=300,
        num_layers=num_layers)

    encoder.load_state_dict(torch.load(
        '/scratch/asw462/models/' + model))
    encoder.cpu()
    torch.save(encoder.state_dict(), '/scratch/asw462/models/CPU_' + model)

