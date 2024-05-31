import torch.nn as nn

class BaseLinear(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 skip: bool = False,
                 batchnorm: bool = False
                 ):
        super().__init__()
        if batchnorm:
            self.module = nn.Sequential(*[nn.Linear(input_channels,output_channels), nn.BatchNorm1d(output_channels)])
        else:
            self.module = nn.Sequential(*[nn.Linear(input_channels,output_channels)])
        self.act = nn.LeakyReLU()
        self.skip = skip
        if skip:
            if input_channels != output_channels:
                self.skip_module = nn.Linear(input_channels, output_channels)
            else:
                self.skip_module = nn.Identity()

    def forward(self, x):
        res =  self.module(x)
        if self.skip:
            return self.act(res + self.skip_module(x))
        else:
            return self.act(res)
