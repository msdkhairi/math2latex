import torch
import torch.nn as nn

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(1)
    
    def forward(self, x):
        x = x + self.encoding[:x.size(0)]
        return self.dropout(x)
    

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=1000, max_w=1000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        self.dropout = nn.Dropout(p=dropout)

        # create self.encoding considering input x as the shape (B, d_model, H, W)
        self.encoding = torch.zeros(max_h, max_w, d_model)
        position_h = torch.arange(0, max_h).unsqueeze(1).float()
        position_w = torch.arange(0, max_w).unsqueeze(1).float()
        div_term_h = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        div_term_w = torch.exp(torch.arange(1, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, :, 0::2] = torch.sin(position_h * div_term_h).unsqueeze(1)
        self.encoding[:, :, 1::2] = torch.cos(position_w * div_term_w).unsqueeze(0)
        self.encoding = self.encoding.permute(2, 0, 1)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(2), :x.size(3)]
        return self.dropout(x)