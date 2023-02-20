import torch
from networks.unet import UNet
from networks.unet_attention import UNet_Attention
from networks.keypoint import Keypoint
from networks.softmax_matcher import SoftmaxMatcher
from networks.svd import SVD

class UnderTheRadar(torch.nn.Module):
    """
        This model computes a 3x3 Rotation matrix and a 3x1 translation vector describing the transformation
        between two radar scans. This transformation can be used for odometry or metric localization.
        It is intended to be an implementation of Under the Radar (Barnes and Posner, 2020)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpuid = config['gpuid']
        self.unet = UNet(config)
        self.keypoint = Keypoint(config)
        self.softmax_matcher = SoftmaxMatcher(config)
        self.svd = SVD(config)
        self.unet_attn = UNet_Attention(config)

    def forward(self, batch):
        data = batch['data'].to(self.gpuid)
        # data_size = data.size()
        # detector_scores, weight_scores, desc = self.unet(data)
        detector_scores, weight_scores, desc = self.unet_attn(data)

        # detector_scores_size = detector_scores.size()
        # weight_scores_size = weight_scores.size()
        # desc_size = desc.size()

        keypoint_coords, keypoint_scores, keypoint_desc = self.keypoint(detector_scores, weight_scores, desc)

        # keypoint_coords_size = keypoint_coords.size()
        # keypoint_scores_size = keypoint_scores.size()
        # keypoint_desc_size = keypoint_desc.size()

        pseudo_coords, match_weights, kp_inds = self.softmax_matcher(keypoint_scores, keypoint_desc, weight_scores, desc)

        # pseudo_coords_size = pseudo_coords.size()
        # match_weights_size = match_weights.size()

        src_coords = keypoint_coords[kp_inds]

        # src_coords_size = src_coords.size()

        R_tgt_src_pred, t_tgt_src_pred = self.svd(src_coords, pseudo_coords, match_weights)

        # R_tgt_src_pred_size = R_tgt_src_pred.size()
        # t_tgt_src_pred_size = t_tgt_src_pred.size()

        return {'R': R_tgt_src_pred, 't': t_tgt_src_pred, 'scores': weight_scores,
                'src': src_coords, 'tgt': pseudo_coords, 'match_weights': match_weights, 'dense_weights': weight_scores}
