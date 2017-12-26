import models.rnn_classifier
import torch
import os


dirs = ["sweep_1124013946",
        "sweep_1124_early_stop_day4",
        "sweep_121804",
        "sweep_1106002002",
        "sweep_1115230657",
        "sweep_1116160716",
        "sweep_1124_early_stop_day2.5",
        "sweep_121800"]

for dir in dirs:
    for model in os.listdir("/scratch/asw462/models/" + dir + "/"):

        h_size = int(filter(lambda s: s.startswith("h_size"), model.split("-"))[0].replace("h_size", ""))
        num_layers = int(filter(lambda s: s.startswith("num_layers"), model.split("-"))[0].replace("num_layers", ""))

        encoder = models.rnn_classifier.ClassifierPooling(
            hidden_size=h_size,
            embedding_size=300,
            num_layers=num_layers)

        print '/scratch/asw462/models/' + dir + "/" + model

        encoder.load_state_dict(torch.load(
            '/scratch/asw462/models/' + dir + "/" + model))
        encoder.cpu()
        torch.save(encoder.state_dict(), '/scratch/asw462/models/' + dir + '/CPU_' + model)

