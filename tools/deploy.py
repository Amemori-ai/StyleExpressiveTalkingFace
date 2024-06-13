"""Deploy Talking Face Model.  input: landmark array, style_space latent """
import os 
import sys 
sys.path.insert(0, os.getcwd()) 
current_path = os.getcwd()
import torch 
import yaml 
import click 
import tqdm
import shutil
import pickle
import cv2
import copy
import shutil

import numpy as np 

from itertools import chain

from easydict import EasyDict as edict 
from TalkingFace.aligner import offsetNet, size_of_alpha, alphas, \
                                update_lip_region_offset, logger, update_region_offset
                                
from TalkingFace.ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.train import stylegan_path
from TalkingFace.ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.equivalent_decoder import EquivalentStyleSpaceDecoder
from TalkingFace.equivalent_offset import fused_offsetNet

norm = lambda x: ((x - x.min(axis = (0,1), keepdims = True)) / (x.max(axis = (0,1), keepdims = True) - x.min(axis = (0,1), keepdims = True)) - 0.5) * 2 
def onnx_infer(
               onnx_path: str,
               *args
              ):
    import onnxruntime as ort 
    def parse_inputs(_inputs):
        if isinstance(_inputs, list):
            for i, x in enumerate(_inputs):
                _inputs[i] = parse_inputs(x)
            return _inputs
        elif isinstance(_inputs, dict):
            for k, v in _inputs:
                _inputs[k] = parse_inputs(v)
            return _inputs
        elif isinstance(_inputs, torch.Tensor):
            return _inputs.cpu().numpy()
    _session = ort.InferenceSession(onnx_path)
    values = [parse_inputs(args[0])] + [parse_inputs(x) for x in args[1]]
    return _session.run(None, dict(zip([x.name for x in _session.get_inputs()], values)))

class TalkingFaceModel(torch.nn.Module):
    """Deploy Model.
    """
    def __init__(self,
                 net_config : str,
                 net_weight_path : str,
                 decoder_path: str
                ):
        super().__init__()

        net = offsetNet(net_config.in_channels * 2, 
                        size_of_alpha, 
                        net_config.depth,
                        base_channels = 512 if not hasattr(net_config, "base_channels") else net_config.base_channels , \
                        dropout = False if not hasattr(net_config, "dropout") else net_config.dropout, \
                        batchnorm = True if not hasattr(net_config, "batchnorm") else net_config.batchnorm, \
                        skip = False if not hasattr(net_config, "skip") else net_config.skip, \
                        norm_type = 'linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
                        )
        net.load_state_dict(torch.load(net_weight_path)['weight'])
        net.eval()
        #net = fused_offsetNet(copy.deepcopy(net))

        decoder = EquivalentStyleSpaceDecoder(stylegan_path, to_resolution = 512)
        decoder.load_state_dict(torch.load(decoder_path), False)
        decoder.eval()

        self.net = fused_offsetNet(copy.deepcopy(net))
        
        self.decoder = decoder

        #self.selected_landmark = torch.from_numpy(np.load(config.selected_path)).float()

    def forward(self, x, latent):
        offset = self.net(x)
        ss_updated = update_lip_region_offset(latent, offset)
        image = self.decoder(ss_updated)
        return torch.nn.functional.interpolate(image, (512,512))
    
def deploy(
            exp_name: str,
            decoder_path: str,
            to_path: str
          ):
    
    #assert os.path.isdir(to_path), "to_path expected is directory."
    
    to_path = os.path.join(to_path, exp_name)
    os.makedirs(to_path, exist_ok = True)
    
    to_path_model = os.path.join(to_path, 'model.pt') 
    to_path_landmark = os.path.join(to_path, 'landmark.npy') 
    to_path_latents = os.path.join(to_path, 'latents.pkl') 

    net_config_path = os.path.join(current_path, 'scripts', exp_name, 'config.yaml')
    net_snapshots_path = os.path.join(current_path, 'results', exp_name, 'snapshots', 'best.pth')

    with open(net_config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    model = TalkingFaceModel(config.net, net_snapshots_path, decoder_path)
    model.eval()
    device = 'cpu'
    model.to(device)
    _input = torch.randn(1, 20, 2).to(device)
    #index = 1
    latent = torch.randn(1,18,512).to(device)
    style_space = model.decoder.get_style_space(latent)
    module = torch.jit.trace(model, (_input, style_space), check_trace = False) 
    #module = torch.jit.trace(model, (_input, 0), check_trace = False) 
    output_original = model(_input, style_space)
    #output_original = model(_input, 0)
    output = module(_input, style_space)
    diff = torch.abs(output_original - output)
    logger.info(f"max error {diff.max()}, min error {diff.min()}, avg error {diff.mean()}")
    #module.save(to_path_model)
    onnx_model_path = to_path_model.replace('pt', 'onnx')
    torch.onnx.export(model, (_input, style_space), onnx_model_path, verbose = True, opset_version = 15, do_constant_folding = True)
    logger.info("get landmark.")

    output_onnx = onnx_infer(onnx_model_path, _input, style_space)
    diff = np.abs(output_original.detach().cpu().numpy() - output_onnx)
    logger.info(f"max error {diff.max()}, min error {diff.min()}, avg error {diff.mean()}")
    _, _, _, selected_id = torch.load(os.path.join(current_path, config.data[0].dataset.id_path))
    landmarks = np.load(os.path.join(current_path, config.data[0].dataset.ldm_path))
    offsets = norm((landmarks - landmarks[selected_id - 1: selected_id, ...])[:, 48:68, ...])
    #np.save(to_path_landmark, landmark)
    shutil.copy(os.path.join(current_path, config.data[0].dataset.id_landmark_path), to_path_landmark)

    logger.info("get style space latents.")
    attributes = torch.load(os.path.join(current_path, config.attr_path))
    pose_latents = torch.load(os.path.join(current_path, config.pose_path))
    n = len(attributes)
    p_bar = tqdm.tqdm(range(n))
    to_save_list = []
    model.to("cuda:0")
    for i in p_bar:
        attribute = attributes[i]
        w_plus_with_pose = pose_latents[i]
        style_space_latent = model.decoder.get_style_space(w_plus_with_pose)
        ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1][size_of_alpha:]).reshape(1, -1).to('cuda:0'), [8, len(alphas)])
        #ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1]).reshape(1, -1).to('cuda:0'), [0, len(alphas)])

        if i < 0:
            image_tensor = model(torch.from_numpy(offsets[i:i+1]).to(device), ss_updated)
            image = image_tensor.detach().cpu().squeeze(0).permute((1,2,0)).numpy()
            image = (image + 1) * 0.5
            cv2.imwrite(f"{i + 1}.jpg", image[...,::-1] * 255.0)

        if len(to_save_list) == 0:
            numpy_arrays = []
            for index, z in enumerate(ss_updated):
                z = z.detach().cpu().numpy()
                _,c = z.shape
                empty_array = np.empty((n,c), dtype = z.dtype)
                empty_array[i:i+1, :] = z
                to_save_list.append(empty_array)
        else:
            for index, z in enumerate(ss_updated):
                z = z.detach().cpu().numpy()
                to_save_list[index][i:i+1, :] = z

    with open(to_path_latents, 'wb') as f:
        pickle.dump(to_save_list, f, protocol = 4)

@click.command()
@click.option('--exp_name')
@click.option('--decoder_path')
@click.option('--to_path')
def _invoker_deploy(
                    exp_name,
                    decoder_path,
                    to_path
                   ):
    return deploy(exp_name, decoder_path, to_path)

if __name__ == '__main__':
    _invoker_deploy()
    


