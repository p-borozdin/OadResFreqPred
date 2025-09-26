import torch
from torch.autograd import Variable


class LSTM(torch.nn.Module):
    def __init__(self, input_size : int, hidden_size : int, activation_func : torch.nn.Module, hidden_layer_size : int, device):
        assert input_size == 1 or input_size == 2, f"Expected input_size = 1 for Temp and 2 for (Temp, gradTemp), but got {input_size}"

        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func = activation_func
        self.hidden_layer_size = hidden_layer_size
        self.device = device

        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, batch_first=True, device=self.device)
        self.fc1 = torch.nn.Linear(self.hidden_size, self.hidden_layer_size)
        self.fc2 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.fc3 = torch.nn.Linear(self.hidden_layer_size, 1)

    def forward(self, x):
        # x.size(0) - batch_size
        assert x.size(2) == self.input_size, f"x.size(2) must be {self.input_size}, but got {x.size(2)}"

        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(self.device)
        c0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(self.device)

        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output[:, -1, :] # Get the last prediction (for the last t)

        out = self.activation_func(self.fc1(output))
        out = self.activation_func(self.fc2(out))
        out = self.fc3(out)

        return out
