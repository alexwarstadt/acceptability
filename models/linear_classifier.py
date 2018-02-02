from torch import nn


class LinearClassifier(nn.Module):
    """
    A basic linear classifier for acceptability judgments.
    Input sentence embedding (with size = 2*encoding_size,
    factor of 2 comes from bidirectional LSTM encoding)
    Hidden layer (encoding 2 * encoding_size * hidden_size)
    Output layer (hidden_size * 1)
    """
    def __init__(self, hidden_size, encoding_size):
        super(LinearClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.enc2h = nn.Linear(2 * encoding_size, self.hidden_size)
        self.h20 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, sentence_vecs):
        hidden = self.tanh(self.enc2h(sentence_vecs))
        out = self.sigmoid(self.h20(hidden))
        return out
