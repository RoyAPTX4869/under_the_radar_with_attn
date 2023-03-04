import torch
from networks.unet import UNet
from networks.keypoint import Keypoint
from positional_embedding import PositionEmbeddingCoordsSine


class UnetTransformer(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.pos_embedding = PositionEmbeddingCoordsSine(n_dim = 2, n_model=256)



    def forward(self, batch):
        data = batch['data'].to(self.gpuid)
        detector_scores, weight_scores, desc = self.unet(data)

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)


