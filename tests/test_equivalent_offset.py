import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import pytest
import numpy as np
import copy

from DeepLog import logger

from TalkingFace.aligner import offsetNet
from TalkingFace.equivalent_offset import fused_offsetNet
from TalkingFace.ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.train import stylegan_path
from TalkingFace.ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.equivalent_decoder import EquivalentStyleSpaceDecoder
from TalkingFace.ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.decoder import StyleSpaceDecoder

@pytest.mark.offset
def test_equivalent_offset_net():

    original_net = offsetNet(40, 21, depth = 1, norm_type = "minmax_constant")
    original_net.load_state_dict(torch.load(os.path.join(os.getcwd(),"..", "results/exp062/snapshots/best.pth"))['weight'])
    fuse_net = fused_offsetNet(copy.deepcopy(original_net))
    fuse_net.eval()
    original_net.eval()
    input_tensor = torch.randn(1, 20, 2)
    output_1 = original_net(input_tensor)
    output_2 = fuse_net(input_tensor)
    diff = torch.abs(output_1 - output_2)
    print(torch.min(diff), torch.max(diff), torch.mean(diff))

@pytest.mark.decoder
def test_equivalent_decoder():
    
    decoder_path = os.path.join(os.getcwd(), "ExpressiveVideoStyleGanEncoding/results/pivot_024/snapshots/350.pth")
    device = "cuda:0"

    decoder = StyleSpaceDecoder(stylegan_path, to_resolution = 512)
    decoder.load_state_dict(torch.load(decoder_path), False)
    decoder.eval()

    equivalent_decoder = EquivalentStyleSpaceDecoder(stylegan_path, to_resolution = 512)
    equivalent_decoder.load_state_dict(torch.load(decoder_path), False)
    equivalent_decoder.eval()

    _input = torch.randn(1, 20, 2).to(device)
    latent = torch.randn(1,18,512).to(device)
    style_space = decoder.get_style_space(latent)

    decoder_image = decoder(style_space)
    equivalent_decoder_image = equivalent_decoder(style_space)

    diff = np.abs(decoder_image.detach().cpu().numpy() - equivalent_decoder_image.detach().cpu().numpy())
    logger.info(f"max error {diff.max()}, min error {diff.min()}, avg error {diff.mean()}")

