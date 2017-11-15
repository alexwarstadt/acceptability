import models.rnn_classifier
import torch

encoder = models.rnn_classifier.ClassifierPooling(
    hidden_size=748,
    embedding_size=300,
    num_layers=3)

encoder.load_state_dict(torch.load(
    '/scratch/asw462/models/sweep_1106235815_rnn_classifier_pooling_14-lr0.00088-h_size748-datadiscriminator-num_layers3'))
encoder.cpu()
torch.save(encoder.state_dict(),
           '/scratch/asw462/models/CPU_sweep_1106235815_rnn_classifier_pooling_14-lr0.00088-h_size748-datadiscriminator-num_layers3')

