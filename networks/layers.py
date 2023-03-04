"""
    Parts of the U-Net model
    Code from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_indices
# from positional_embedding import PositionEmbeddingCoordsSine

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            d = int(pow(2, np.floor(np.log(in_channels) / np.log(2))))
            self.up = nn.ConvTranspose2d(d, d, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output 1x1 convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Self_Attn_Layer (nn.Module):
    def __init__(self, config, feat_dim, n_head=8,dropout=0.1, layer_norm=True):
        super().__init__()
        self.mulihead_attn = nn.MultiheadAttention(feat_dim, n_head, dropout=dropout)
        self.attn_weights = None
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.norm = layer_norm

    def forward(self, feat):
        size = feat.size()
        # print(size[1])
        feat_3d = feat.view(size[0], size[1], -1)
        if self.norm:
            feat_norm = self.layer_norm(feat_3d.permute((0,2,1))).permute((0,2,1))      
            q = k = v = feat_norm.permute((2, 0, 1)) 
        else:
            q = k = v = feat_3d.permute((2, 0, 1))
            
        attn_out, _= self.mulihead_attn(q, k, v, need_weights=True)
        # self.attn_weights = attn_weights
        attn_out_trans = attn_out.permute((1,2,0)).view(size)

        return attn_out_trans

class Cross_Attn_Layer (nn.Module):
    def __init__(self, config, feat_dim, n_head=8,dropout=0.1, layer_norm=True):
        super().__init__()
        self.mulihead_attn = nn.MultiheadAttention(feat_dim, n_head, dropout=dropout)
        self.attn_weights = None
        self.norm = layer_norm
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.config = config
        self.window_size = config['window_size']

    def forward(self,feat):
        size = feat.size()
        # print(size[1])
        feat_3d = feat.view(size[0], size[1], -1)

        BW = feat.size(0)
        Batch_size = int (BW / self.window_size)
        src_ids, tgt_ids = get_indices(Batch_size, self.window_size)
        src_feat = feat_3d[src_ids]
        tgt_feat = feat_3d[tgt_ids]

        assert len(src_ids) == len(tgt_ids), "input data imgs are not in paired"

        #### get q, k, v for cross-attn
        if self.norm:
            src_feat_norm = self.layer_norm(src_feat.permute((0,2,1))).permute((0,2,1)) #(BW/2, C, H*W)
            tgt_feat_norm = self.layer_norm(tgt_feat.permute((0,2,1))).permute((0,2,1)) #(BW/2, C, H*W)

            src_q = src_feat_norm.permute((2, 0, 1))                                    #(H*W, BW/2, C)
            src_v  = src_k = tgt_feat_norm.permute((2, 0, 1))                           #(H*W, BW/2, C)

            tgt_q = tgt_feat_norm.permute((2, 0, 1))                                    #(H*W, BW/2, C)
            tgt_v  = tgt_k = src_feat_norm.permute((2, 0, 1))                           #(H*W, BW/2, C)
        else:
            src_q = src_feat.permute((2, 0, 1))
            src_v = src_k = tgt_feat.permute((2, 0, 1))
            tgt_q = tgt_feat.permute((2, 0, 1))
            tgt_v = tgt_k = src_feat.permute((2, 0, 1))

        #### src-to-tgt cross attn
        cross_attn_out_src, _ = self.mulihead_attn(src_q, src_k, src_v)                 #(H*W, BW/2, C)
        cross_attn_out_src = cross_attn_out_src.permute((1,2,0))                        #(BW/2, C, H*W)
        cross_attn_out_src = cross_attn_out_src.view(len(src_ids), size[1], size[2], size[3])   #(BW/2, C, H, W)    

        #### tgt-to-src cross attn
        cross_attn_out_tgt, _ = self.mulihead_attn(tgt_q, tgt_k, tgt_v)                 #(H*W, BW/2, C)
        cross_attn_out_tgt = cross_attn_out_tgt.permute((1,2,0))                        #(BW/2, C, H*W)
        cross_attn_out_tgt = cross_attn_out_tgt.view(len(tgt_ids), size[1], size[2], size[3])   #(BW/2, C, H, W) 

        x = len(src_ids)
        for i in range(0, len(src_ids)):
            if i == 0:
                cross_attn_out_trans = torch.cat((cross_attn_out_src[i], cross_attn_out_tgt[i]), 0)
                cross_attn_out_trans = cross_attn_out_trans.view(2,size[1],size[2],size[3])     #(2, C, H, W) 
            else: 
                # print(cross_attn_out_src[i].size())
                cross_attn_out_trans = torch.cat((cross_attn_out_trans,                         #(BW, C, H, W) 
                                                cross_attn_out_src[i].view(1,size[1],size[2],size[3]),
                                                cross_attn_out_tgt[i].view(1,size[1],size[2],size[3])),
                                                 0)
        # cross_attn_out_trans = cross_attn_out_src.permute((1,2,0))[0]
        # self.attn_weights = (attn_weights_src, attn_weights_tgt)
        return cross_attn_out_trans

class Self_Attn_Block (nn.Module):
    def __init__(self, config, feat_dim, feat_feedforward, n_head=8,dropout=0.1, layer_norm=False, use_pos_emb=False):
        super().__init__()
        self.config = config
        self.attn_weights = None
        self.norm = layer_norm
        
        self.mulihead_attn_self = Self_Attn_Layer(config, feat_dim, n_head=n_head,dropout=dropout, layer_norm=False)

        self.layer_norm = nn.LayerNorm(feat_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(feat_dim, feat_feedforward)
        self.linear2 = nn.Linear(feat_feedforward, feat_dim)
        self.activation = F.relu

    def forward(self, feat):
        #### multihead self attention
        self_attn = self.mulihead_attn_self.forward(feat)
        self_attn = self_attn + self.dropout1(feat)

        #### Position-wise feedforward
        self_attn_2 = self.layer_norm(self_attn.permute((0,2,3,1))).permute((0,3,1,2))

        size = self_attn_2.size()
        self_attn_2 = self_attn_2.view(size[0],size[1],-1).permute(0,2,1)   #(BW, H'*W', C)
        self_attn_2 = self.linear2(self.dropout2(self.activation(self.linear1(self_attn_2))))
        self_attn_2 = self_attn_2.permute(0,2,1).view(size) #(BW, C, H', W')

        return self_attn_2


class Cross_Attn_Block (nn.Module):
    def __init__(self, config, feat_dim, feat_feedforward, n_head=8,dropout=0.1, layer_norm=False ):
        super().__init__()
        self.config = config
        self.attn_weights = None
        self.norm = layer_norm
        
        self.mulihead_attn_cross = Cross_Attn_Layer(config, feat_dim, n_head=n_head,dropout=dropout, layer_norm=False)

        self.layer_norm = nn.LayerNorm(feat_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(feat_dim, feat_feedforward)
        self.linear2 = nn.Linear(feat_feedforward, feat_dim)
        self.activation = F.relu

    def forward(self, feat):
        #### multihead cross attention
        cross_attn = self.mulihead_attn_cross.forward(feat)
        cross_attn = cross_attn + self.dropout1(feat)

        #### Position-wise feedforward
        cross_attn_2 = self.layer_norm(cross_attn.permute((0,2,3,1))).permute((0,3,1,2))

        size = cross_attn_2.size()
        cross_attn_2 = cross_attn_2.view(size[0],size[1],-1).permute(0,2,1)   #(BW, H'*W', C)
        cross_attn_2 = self.linear2(self.dropout2(self.activation(self.linear1(cross_attn_2))))
        cross_attn_2 = cross_attn_2.permute(0,2,1).view(size) #(BW, C, H', W')

        return cross_attn_2


