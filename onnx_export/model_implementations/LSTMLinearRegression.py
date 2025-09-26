"""Модуль описывает... (описываем модуль)
"""
import torch
from torch.autograd import Variable


class LSTM(torch.nn.Module):
    """Краткое описание класса
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            device: torch.device
            ):
        """_summary_

        Args:
            input_size (int): _description_
            hidden_size (int): _description_
            device (torch.device): _description_
        """
        assert input_size in (1, 2), \
            f"Expected input_size = 1 for Temp and 2 for (Temp, gradTemp), " \
            f"but got {input_size}"

        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            batch_first=True,
            device=self.device
            )
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_features=self.hidden_size, out_features=1)
        # Инициализация архитектуры (torch list'a/dict'a)
        # self.init_model

    # def init_model(self):
        # """_summary_
        # """

        # Инициализация torch list'om (torch dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # x.size(0) - batch size
        assert x.size(2) == self.input_size, f"x.size(2) must be {self.input_size}, but got {x.size(2)}"

        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(self.device)
        c0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to(self.device)

        '''
        FIXME: there must be 
        out = self.relu(output)
        out = self.fc(out)
        '''

        output, (hn, _) = self.lstm(x, (h0, c0))
        # hn = hn.view((x.size(0), self.hidden_size))
        out = output[:, -1, :]  # Get the last LSTM prediction
        out = self.relu(hn)
        out = self.fc(out)

        return out
