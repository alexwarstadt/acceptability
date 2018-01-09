import models.acceptability_cl
import models.cbow
import torch
import os


dirs = ["sweep_1229145144",
        "sweep_1227010500",
        "sweep_1229132558"]

# for dir in dirs:
#     for model in os.listdir("/scratch/asw462/models/" + dir + "/"):
#
#         h_size = int(filter(lambda s: s.startswith("h_size"), model.split("-"))[0].replace("h_size", ""))
#         enc_size = int(filter(lambda s: s.startswith("h_size"), model.split("-"))[1].replace("h_size", ""))
#
#         encoder = models.acceptability_cl.Classifier(
#             hidden_size=h_size,
#             encoding_size=enc_size)
#
#         print '/scratch/asw462/models/' + dir + "/" + model
#
#         try:
#             encoder.load_state_dict(torch.load(
#                 '/scratch/asw462/models/' + dir + "/" + model))
#             encoder.cpu()
#             torch.save(encoder.state_dict(), '/scratch/asw462/models/' + dir + '/CPU_' + model)
#         except KeyError:
#             pass




dirs = ["sweep_1229005541"]
for dir in dirs:
    for model in os.listdir("/scratch/asw462/models/" + dir + "/"):

        h_size = int(filter(lambda s: s.startswith("h_size"), model.split("-"))[0].replace("h_size", ""))
        max_pool = filter(lambda s: s.startswith("max_pool"), model.split("-"))[0].replace("max_pool", "") == "1"

        encoder = models.cbow.Classifier(
            hidden_size=h_size,
            input_size=300,
            max_pool=max_pool
        )

        print '/scratch/asw462/models/' + dir + "/" + model

        try:
            encoder.load_state_dict(torch.load(
                '/scratch/asw462/models/' + dir + "/" + model))
            encoder.cpu()
            torch.save(encoder.state_dict(), '/scratch/asw462/models/' + dir + '/CPU_' + model)
        except KeyError:
            pass