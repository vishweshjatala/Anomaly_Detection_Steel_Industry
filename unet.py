import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_cols):
        super().__init__()
        self.in_dim = num_cols
        self.dims = [32,64,128]
        self.in_proj = nn.Linear(self.in_dim, self.dims[0], bias=False)
        self.encoders = nn.ModuleList()
        _in = self.dims[0]
        for i in range(1,len(self.dims)):
            self.encoders.append(self._create_enc(_in, self.dims[i]))
            _in = self.dims[i]

        self.decoders = nn.ModuleList()
        outs = self.dims[::-1]
        _out = outs[0]
        for i in range(1,len(outs)):
            self.decoders.append(self._create_dec(_out*2, outs[i]))
            _out = outs[i]
        self.out_proj = nn.Linear(outs[-1], self.in_dim, bias=False)

    def _create_enc(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 3),
            nn.GELU()
        )
    def _create_dec(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose1d(in_dim, out_dim, 3),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, 1)
        )

    def forward(self, x):
        """
        x: b,t,c
        """
        x = self.in_proj(x)
        x = x.transpose(-1,-2)
        residuals = []
        for encoder in self.encoders:
            x = encoder(x)
            residuals.append(x)

        residuals = residuals[::-1]
        for i,decoder in enumerate(self.decoders):
            x = torch.cat([x,residuals[i]],dim=1)
            x = decoder(x)
        x = self.out_proj(x.transpose(-1,-2))

        return x