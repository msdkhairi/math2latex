import torch
import torch.nn as nn

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding1D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.dropout(self.encoding[:, :x.size(1)].detach())
    

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        self.encoding = torch.zeros(max_len, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, :, 0::2] = torch.sin(position * div_term)
        self.encoding[:, :, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    
    def forward(self, x):
        print("self.encoding[:, :x.size(1), :x.size(2)].detach().shape: ", self.encoding[:, :x.size(1), :x.size(2)].detach().shape)
        return x + self.dropout(self.encoding[:, :x.size(1), :x.size(2)].detach())