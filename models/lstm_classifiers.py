from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers,
                 reduction_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.reduction_size = reduction_size
        self.ih2h = nn.LSTM(embedding_size, hidden_size,
                            num_layers=num_layers,
                            bidirectional=True)
        self.h2r = nn.Linear(2 * hidden_size, reduction_size)
        self.r2o = nn.Linear(reduction_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, hidden_states):
        o, hc = self.ih2h(x, hidden_states)
        reduction = self.sigmoid(self.h2r(o[-1]))
        output = self.sigmoid(self.r2o(reduction))
        return output, reduction


class LSTMPoolingClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers):
        super(LSTMPoolingClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.ih2h = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers,
                            bidirectional=True)
        self.pool2o = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        o, hc = self.ih2h(x, hidden_states)
        pool = nn.functional.max_pool1d(torch.transpose(o, 0, 2), len(input))
        pool = torch.transpose(pool, 0, 2).squeeze()
        output = self.sigmoid(self.pool2o(pool))
        return output, pool
