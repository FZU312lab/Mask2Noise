import torch
from torch import nn
from basic_layers import ConvBlock,CrossAttentionTransformer


class NAM(nn.Module):
    def __init__(self,in_channels,dim,num_queries,num_heads=8,use_query_update=True,use_spatial_gate=True):
        super().__init__()
        self.dim = dim
        self.num_queries=num_queries
        self.use_query_update = use_query_update
        self.use_spatial_gate = use_spatial_gate
        # local feature extraction
        self.conv = ConvBlock(in_channels,dim)
        if use_query_update:
            self.cat = CrossAttentionTransformer(dim=dim,num_heads=num_heads)
        # Query projection
        self.query_projection = nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))

        # Spatial gate predictor
        if use_spatial_gate:
            self.mask_predictor = nn.Conv2d(dim,1,kernel_size=1)

    def forward(self,feature,query_state,pos_encoding):
        local_feature = self.conv(feature)
        B,C,H,W = local_feature.shape
        tokens = local_feature.flatten(2).transpose(1,2)
        # [B,HW,C]
        tokens = pos_encoding(tokens,H,W)

        if self.use_query_update:
            query_state = self.cat(query_state,tokens)

        if self.use_spatial_gate:
            projected_query = self.query_projection(query_state)
            correlation = projected_query.mean(dim=1).unsqueeze(-1).unsqueeze(-1) * local_feature
            mask = torch.sigmoid(self.mask_predictor(correlation))
            feature_out = (local_feature *mask)
        else:
            feature_out = local_feature

        return (feature_out,query_state)
