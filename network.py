import torch
from torch import nn
from basic_layers import *
from NAM import *
from reconstruction_head import *

class Mask2Noise(nn.Module):
    def __init__(
        self,
        in_channels=1,
        dim=64,
        num_stages=4,
        num_queries=64,
        num_heads=8,
        query_update=True,
        spatial_gate=True,
        use_qfrr=True
    ):
        super().__init__()
        self.dim = dim
        self.num_stages=num_stages
        self.num_queries=num_queries
        self.query_update=query_update
        self.spatial_gate=spatial_gate
        self.use_qfrr=use_qfrr
        self.query_state = nn.Parameter(torch.randn(num_queries,dim))
        self.pos_encoding = PositionalEncoding(dim)
        stages=[]
        for i in range(num_stages):
            stages.append(NAM(in_channels=in_channels if i==0 else dim,
                              dim=dim,
                              num_queries=num_queries,
                              num_heads=num_heads,
                              use_query_update=query_update,
                              use_spatial_gate=spatial_gate))
        self.encoder = nn.ModuleList(stages)
        if use_qfrr:
            self.qfrr = QFRR(dim=dim)
        else:
            self.residual_head = nn.Sequential(
                nn.Conv2d(dim,dim,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim,1,kernel_size=3,padding=1)
            )

    def forward(self,x,return_query=False,return_features=False):
        B = x.shape[0]
        query_state = (self.query_state.unsqueeze(0).expand(B,-1,-1))
        feature = x
        stage_features=[]
        for stage in self.encoder:
            feature, query_state = stage(feature,query_state,self.pos_encoding)
            if return_features:
                stage_features.append(feature)
        if self.use_qfrr:
            residual = self.qfrr(feature,query_state)
        else:
            residual = self.residual_head(feature)

        denoised = x - residual
        outputs=[denoised]
        if return_query:
            outputs.append(query_state)

        if return_features:
            outputs.append(stage_features)

        if len(outputs)==1:
            return denoised
        else:
            return tuple(outputs)
