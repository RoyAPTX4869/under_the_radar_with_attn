import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_indices
from positional_embedding import PositionEmbeddingCoordsSine


class Self_Attn_Layer (nn.Module):
    def __init__(self, config, feat_dim, n_head=8,dropout=0.1):
        super().__init__()
        self.mulihead_attn = nn.MultiheadAttention(feat_dim, n_head, dropout=dropout)
        self.attn_weights = None
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.pos_emb = PositionEmbeddingCoordsSine(n_dim = 2, n_model=248)

    def forward(self, feat, pos):
        """ 
        Args:
            feat (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches), and Channel is set to 248
            pos (torch.tensor): (Batch_size*Window_size,  Num_patches,  2)
        Returns:
            attn_out_trans (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches) 
        """
        size = feat.size()
        BW = size[0]

        # layer norm
        feat_norm = self.layer_norm(feat)

        # pose-embedding
        # pose_emb = torch.zeros_like(feat)
        # for i in range(BW):
        #     pose_emb = self.pos_emb(pos[i])
        #     feat_norm_emd = pose_emb + feat_norm
        pose_emb = self.pos_emb(pos)
        feat_norm_emd = pose_emb.permute((0,2,1)) + feat_norm           #feat_norm_emd: (Batch_size*Window_size,  Channel,  Num_patches)

        ## attention 
        q = k = v = torch.permute(feat_norm_emd,(2, 0, 1)) 
        attn_out, attn_weights= self.mulihead_attn(q, k, v, need_weights=True)
        self.attn_weights = attn_weights
        attn_out_trans = attn_out.permute((1,2,0)).view(size)

        return attn_out_trans

class Cross_Attn_Layer (nn.Module):
    def __init__(self, config, feat_dim, n_head=8,dropout=0.1):
        super().__init__()
        self.mulihead_attn = nn.MultiheadAttention(feat_dim, n_head, dropout=dropout)
        self.attn_weights = None
        self.pos_emb = PositionEmbeddingCoordsSine(n_dim = 2, n_model=248)
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.config = config
        self.window_size = config['window_size']

    def forward(self, feat, pos):
        """ 
        Do the attention processes which includs pre-norm & positional embedding & multihead attention 
        Args:
            feat (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches), and Channel is set to 248
            pos (torch.tensor): (Batch_size*Window_size,  Num_patches,  2)
        Returns:
            attn_out_trans (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches) 
        """
        size = feat.size()

        #### split src and tgt fig
        BW = feat.size(0)
        Batch_size = int (BW / self.window_size)
        src_ids, tgt_ids = get_indices(Batch_size, self.window_size)
        src_feat = feat[src_ids]
        tgt_feat = feat[tgt_ids]
        

        assert len(src_ids) == len(tgt_ids), "input data imgs are not in paired"     
        #### layer norm
        src_feat_norm = self.layer_norm(src_feat)
        tgt_feat_norm = self.layer_norm(tgt_feat)

        #### pose-embedding
        pos_emb = self.pos_emb(pos)
        src_pos_emb = pos_emb[src_ids]
        tgt_pos_emb = pos_emb[tgt_ids]
        src_feat_norm_emb = src_feat_norm + src_pos_emb         # (Batch_size*Window_size,  Channel,  Num_patches)
        tgt_feat_norm_emb = tgt_feat_norm + tgt_pos_emb         # (Batch_size*Window_size,  Channel,  Num_patches)

        #### get q, k, v for cross-attn
        src_q = src_feat_norm_emb.permute((2, 0, 1))
        src_v  = src_k = tgt_feat_norm_emb.permute((2, 0, 1))
        tgt_q = src_feat_norm_emb.permute((2, 0, 1))
        tgt_v  = tgt_k = tgt_feat_norm_emb.permute((2, 0, 1))   # (Num_patches, Batch_size*Window_size, Channel)
 

        #### src-to-tgt cross attn
        cross_attn_out_src, attn_weights_src= self.mulihead_attn(src_q, src_k, src_v, need_weights=True)
        cross_attn_out_src = cross_attn_out_src.permute((1,2,0))
        cross_attn_out_src = cross_attn_out_src.view(len(src_ids), size[1], size[2], size[3])

        #### tgt-to-src cross attn
        cross_attn_out_tgt, attn_weights_tgt= self.mulihead_attn(tgt_q, tgt_k, tgt_v, need_weights=True)
        cross_attn_out_tgt = cross_attn_out_tgt.permute((1,2,0))
        cross_attn_out_tgt = cross_attn_out_tgt.view(len(tgt_ids), size[1], size[2], size[3])

        x = len(src_ids)
        for i in range(0, len(src_ids)):
            if i == 0:
                cross_attn_out_trans = torch.cat((cross_attn_out_src[i], cross_attn_out_tgt[i]), 0)
                cross_attn_out_trans = cross_attn_out_trans.view(2,size[1],size[2],size[3])
            else: 
                # print(cross_attn_out_src[i].size())
                cross_attn_out_trans = torch.cat((cross_attn_out_trans, 
                                                cross_attn_out_src[i].view(1,size[1],size[2],size[3]),
                                                cross_attn_out_tgt[i].view(1,size[1],size[2],size[3])),
                                                 0)
        # cross_attn_out_trans = cross_attn_out_src.permute((1,2,0))[0]
        self.attn_weights = (attn_weights_src, attn_weights_tgt)
        return cross_attn_out_trans

class Self_Attn_Block (nn.Module):
    def __init__(self, config, feat_dim, feat_feedforward, n_head=8,dropout=0.1, layer_norm=False):
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

    def forward(self, feat, pos):
        """ 
        Implement of the transformer encoder which includs attention block & position-wise feedforward 
        Args:
            feat (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches), and Channel is set to 248
            pos (torch.tensor): (Batch_size*Window_size,  Num_patches,  2)
        Returns:
            self_attn_2 (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches) 
        """
        #### multihead self attention
        self_attn = self.mulihead_attn_self.forward(feat, pos)
        self_attn = self_attn + self.dropout1(feat)

        #### Position-wise feedforward
        self_attn_2 = self.layer_norm(self_attn.permute((0,2,3,1))).permute((0,3,1,2))

        size = self_attn_2.size()
        self_attn_2 = self_attn_2.view(size[0],size[1],-1).permute(0,2,1)   #(BW, H'*W', C)
        self_attn_2 = self.linear2(self.dropout2(self.activation(self.linear1(self_attn_2))))
        self_attn_2 = self_attn_2.permute(0,2,1).view(size) #(BW, C, H', W')

        return self_attn_2


class Cross_Attn_Block (nn.Module):
    def __init__(self, config, feat_dim, feat_feedforward, n_head=8,dropout=0.1, layer_norm=False):
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

    def forward(self, feat, pos):
        """ 
        Implement of the transformer encoder which includs attention block & position-wise feedforward 
        Args:
            feat (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches), and Channel is set to 248
            pos (torch.tensor): (Batch_size*Window_size,  Num_patches,  2)
        Returns:
            self_attn_2 (torch.tensor): (Batch_size*Window_size,  Channel,  Num_patches) 
        """
        #### multihead cross attention
        cross_attn = self.mulihead_attn_cross.forward(feat, pos)
        cross_attn = cross_attn + self.dropout1(feat)

        #### Position-wise feedforward
        cross_attn_2 = self.layer_norm(cross_attn.permute((0,2,3,1))).permute((0,3,1,2))

        size = cross_attn_2.size()
        cross_attn_2 = cross_attn_2.view(size[0],size[1],-1).permute(0,2,1)   #(BW, H'*W', C)
        cross_attn_2 = self.linear2(self.dropout2(self.activation(self.linear1(cross_attn_2))))
        cross_attn_2 = cross_attn_2.permute(0,2,1).view(size) #(BW, C, H', W')

        return cross_attn_2

class Decoder_Block(nn.Module):
    def __init__(self, d_embed):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self,feats,coor):
        """
        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """
