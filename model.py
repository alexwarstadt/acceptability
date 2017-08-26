import torch
import torch.nn as nn
from torch.autograd import Variable

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, nonlinearity):
        super(MyLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # self.ih2h = nn.LSTM(input_size, hidden_size)
        # self.h2h = nn.LSTM(input_size, hidden_size, n_layers)

        self.ih2h = nn.LSTMCell(input_size, hidden_size)
        # self.h2h = []
        # for _ in range(n_layers):
        #     self.h2h.append(nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size))
        self.h2h = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.functional.log_softmax
        self.softmax = nn.functional.softmax
        # self.nonlinearity = nonlinearity

    # def forward(self, input, hidden_states):
    #     """version for LSTM cell"""
    #     h, c = self.ih2h(input, hidden_states[0])
    #     next_hiddens = [(h, c)]
    #     for state, layer in zip(hidden_states, self.h2h):
    #         h, c = layer(h, state)
    #         next_hiddens.append((h, c))
    #     output = self.log_softmax(self.h2o(h))
    #     return output, next_hiddens

    def forward(self, input, hidden_states):
        """version for LSTM cell"""
        h, c = self.ih2h(input, hidden_states[0])
        next_hiddens = [(h, c)]
        h, c = self.h2h(h, hidden_states[1])
        next_hiddens.append((h, c))
        output = self.log_softmax(self.h2o(h))
        return output, next_hiddens

    # def forward(self, input, hidden_states):
    #     """version for regular LSTM"""
    #     o, h = self.ih2h(input, hidden_states[0])
    #     next_hiddens = [h]
    #     o, h = self.h2h(h, hidden_states[1:])
    #     next_hiddens.extend(h)
    #     output = self.log_softmax(self.h2o(o))
    #     return output, hidden_states

    # def init_hidden(self, batch_size):
    #     hidden_states = []
    #     for i in range(self.n_layers + 1):
    #         hidden_states.append((Variable(torch.zeros(1, batch_size, self.hidden_size)),
    #                               Variable(torch.zeros((1, batch_size, self.hidden_size)))))
    #     return hidden_states

    def init_hidden(self, batch_size):
        hidden_states = []
        for i in range(self.n_layers + 1):
            hidden_states.append((Variable(torch.zeros(batch_size, self.hidden_size)),
                                  Variable(torch.zeros(batch_size, self.hidden_size))))
        return hidden_states

    def init_hidden_single(self):
        hidden_states = []
        for i in range(self.n_layers + 1):
            hidden_states.append((Variable(torch.zeros(1, self.hidden_size)),
                                  Variable(torch.zeros(1, self.hidden_size))))
        return hidden_states

    def n_params(self):
        return (self.input_size + self.hidden_size) * self.hidden_size + \
            self.n_layers * self.hidden_size * self.hidden_size + \
            self.hidden_size * self.output_size



class RNN(nn.Module):
    """Multilayer but unconnected RNN. i.e. the deep layers are not RNNs in and of themselves, rather the hidden
    state at each staged is passed through a number of linear layers and the result is the hidden state given
    to the next iteration of the sequence"""
    def __init__(self, input_size, hidden_size, output_size, n_layers, nonlinearity, model_type="RNN"):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.model_type = model_type

        self.ih2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2h = []
        for i in range(n_layers):
            self.h2h.append(nn.Linear(hidden_size, hidden_size))
        self.h2o = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.functional.log_softmax
        self.softmax = nn.functional.softmax
        self.nonlinearity = nonlinearity

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.nonlinearity(self.ih2h(combined))
        for hidden_layer in self.h2h:
            hidden = self.nonlinearity(hidden_layer(hidden))
        output = self.log_softmax(self.h2o(hidden))
        return output, hidden

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))

    def init_hidden_single(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def n_params(self):
        return (self.input_size + self.hidden_size) * self.hidden_size + \
            self.n_layers * self.hidden_size * self.hidden_size + \
            self.hidden_size * self.output_size