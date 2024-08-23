import os
import sys
sys.path.insert(0, os.getcwd())
import onnx
import torch
import pytest
import onnx_graphsurgeon as gs

class Net(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        #self.register_parameter("values", torch.nn.parameter.Parameter(torch.randn((channels, 2))))
        self.register_buffer("values",torch.randn((channels, 2)))

    def forward(self, x):
        return x * self.values[:, 0] + self.values[:,1]

@pytest.mark.convert
def test_convert_onnx():
    
    model = Net(512)
    _input = torch.randn((1,512), dtype = torch.float32)
    onnx_model_path = "test.onnx"
    torch.onnx.export( \
                      model, (_input), \
                      onnx_model_path, verbose = True, \
                      opset_version = 15, do_constant_folding = True
                     )

@pytest.mark.surgeon
def test_onnx_surgeon():

    path = "release/man3_s0_20240118_expressive_version/model.onnx"
    
    #path = "test.onnx"
    original_model = onnx.load(path)

    graph = original_model.graph

    # initializer
    for node in graph.initializer:
        print(node.name, type(node))


    graph = gs.import_onnx(original_model)
    onnx_info = graph.tensors()
    for k, v in onnx_info.items():
        print(v)
