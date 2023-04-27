import torch
import torch.nn as nn


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        out = output[:, -1, :]
        out = self.fc(out)
        return out

