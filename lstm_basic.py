import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMBasic(nn.Module):
    def __init__(self,num_cols):
        super().__init__()
        self.num_cols = num_cols
        self.enc1 = nn.LSTM(self.num_cols, 128, num_layers=1, batch_first=True)
        self.enc2 = nn.LSTM(128, 64, num_layers=1, batch_first=True)
        self.enc3 = nn.LSTM(64, 32, num_layers=1, batch_first=True)
        self.ln1  = nn.LayerNorm(32)
        self.proj = nn.Linear(32,32,bias=False)
        self.drop = nn.Dropout(0.1)
        self.dec3 = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.dec2 = nn.LSTM(64, 128, num_layers=1, batch_first=True)
        self.ln2  = nn.LayerNorm(128)
        self.dec1 = nn.Linear(128, self.num_cols, bias=False)
        self.act1 = nn.Identity()
        self.act2 = nn.Identity()

    def forward(self, x):

        x,_ = self.enc1(x)
        x = self.act1(x)
        x,_ = self.enc2(x)
        x = self.act2(x)
        x,_ = self.enc3(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = self.act2(x)
        x = self.proj(x)
        x = self.drop(x)
        x,_ = self.dec3(x)
        x = self.act1(x)
        x,_ = self.dec2(x)
        x = self.act2(x)
        x = self.ln2(x)
        x = self.dec1(x)

        return x