import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SelfAttention(nn.Module):
    def __init__(self, input_dim:int, model_dim:int, device):
        super(SelfAttention, self).__init__()
        self.device = device
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        self.query = nn.Linear(self.input_dim, self.model_dim)
        self.key = nn.Linear(self.input_dim, self.model_dim)
        self.value = nn.Linear(self.input_dim, self.model_dim)
        self.softmax = nn.Softmax(dim=2)
        self.normalization = self.model_dim ** 0.5
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.normalization
        
        attention = self.softmax(scores)
        
        weighted = torch.bmm(attention, values)
        
        return weighted

class LSTM_Block(nn.Module):
    def __init__(self, input_size : int, hidden_size : int, device):
        super(LSTM_Block, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, device=self.device)

    def forward(self, x):
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(self.device)
        c0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(self.device)

        output, (hn, cn) = self.lstm(x, (h0, c0))
        # output = output[:, -1, :] # Get the last prediction (for the last t)
        # return output.view(output.shape[0], 1, output.shape[1])
        return output
    

class LSTM_with_SA(nn.Module):
    def __init__(self, input_size : int, seq_len:int, hidden_size : int, model_sa_dim : int, activation_func : nn.Module, hidden_layer_size : int,  device):
        assert input_size == 1 or input_size == 2, f"Expected input_size = 1 for Temp and 2 for (Temp, gradTemp), but got {input_size}"
        super(LSTM_with_SA, self).__init__()

        self.device = device # GPU/CPU device

        self.input_size = input_size # parameters: 1 for Temp and 2 for (Temp, gradTemp)
        self.seq_len = seq_len # sequence length
        

        self.hidden_size = hidden_size # parameters: Output LSTM features 

        self.model_sa_dim = model_sa_dim # parameters for Self-Attention
        
        self.hidden_layer_size = hidden_layer_size # parameters for FFNN
        self.activation_func = activation_func # activation in FFNN

        self.lstm = LSTM_Block(self.input_size, hidden_size = self.hidden_size, device=self.device)

        self.sa = SelfAttention(input_dim = self.hidden_size, model_dim = self.model_sa_dim, device=self.device)
        
        self.fc1 = nn.Linear(self.seq_len*self.model_sa_dim, self.hidden_layer_size)
        self.fc2 = nn.Linear(self.hidden_layer_size, 1)
        
    def forward(self, x):
        assert x.size(2) == self.input_size, f"x.size(2) must be {self.input_size}, but got {x.size(2)}"
        # LSTM part
        LSTM_output = self.lstm(x)
        # Self-Attention part
        SA_output = self.sa(LSTM_output)
        SA_output = SA_output.view(SA_output.size(0), -1)
        # Feed Forward part
        output = self.activation_func(self.fc1(SA_output))
        output = self.fc2(output)

        return output
