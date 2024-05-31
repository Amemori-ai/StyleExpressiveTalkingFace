import torch
import copy
from .module import BaseLinear

class fused_bn_base_linear(torch.nn.Module):

    def __init__(
                 self,
                 module: torch.nn.Module
                ):
        super().__init__()
        module_seq = module.module
        _first_linear, _bn = module_seq[0], module_seq[1]

        _first_weight = _first_linear.weight
        _first_bias = _first_linear.bias
        out_channels, in_channels = _first_weight.shape

        bn_mean = _bn.running_mean
        bn_var = torch.rsqrt(_bn.running_var + _bn.eps)

        bn_scale = _bn.weight #n, c
        bn_shift = _bn.bias #n, c
        
        #oxi = oxo * oxi

        new_weight = _first_weight * (bn_scale * bn_var).unsqueeze(-1)
        new_bias = (_first_bias - bn_mean) * bn_scale * bn_var + bn_shift

        linear = copy.deepcopy(_first_linear)#torch.nn.Linear(in_channels, out_channels)

        linear.weight = torch.nn.Parameter(new_weight)
        linear.bias = torch.nn.Parameter(new_bias)


        self.module = torch.nn.Sequential(linear)
        self.act = copy.deepcopy(module.act)

    def forward(self, x):
        return self.act(self.module(x))


class fused_offsetNet(torch.nn.Module):
    def __init__(
                  self,
                  module: torch.nn.Module
                ):
        super().__init__()
        
        for i in range(len(module.net)):
            if isinstance(module.net[i],BaseLinear):
                module.net[i] = fused_bn_base_linear(module.net[i]) 

        self.module = module

    def forward(self, x):
        return self.module(x)

