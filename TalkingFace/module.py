import torch.nn as nn

class BaseLinear(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 skip: bool = False,
                 norm: str = None
                 ):
        super().__init__()
        self.module = nn.Sequential(*[nn.Linear(input_channels,output_channels), getattr(nn, norm)(output_channels)])
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

class BaseConv2d(nn.Module):
    def __init__(
                 self,
                 input_channels,
                 output_channels,
                 k,
                 s,
                 p,
                 norm = "BatchNorm2d",
                 act = "ReLU",
                 repeat = 0
                ):
        super().__init__()

        if act == "ReLU":
            args = dict()
        else:
            args = dict(negative_slope = 0.1)

        modules = [
                    nn.Conv2d(input_channels, output_channels, k, s, p),
                    getattr(nn, norm)(output_channels),
                    getattr(nn, act)(**args)
                  ]


        if repeat > 0:
            modules += [
                        nn.Conv2d(output_channels, output_channels, k, s, p),
                        getattr(nn, norm)(output_channels),
                        getattr(nn, act)(**args)
                       ]

        self.base_conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.base_conv(x)

class Flatten(nn.Module):
    def forward(
                self,
                x
               ):
        n = x.shape[0]
        return x.reshape(n, -1)



