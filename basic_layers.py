import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)


class PositionalEncoding(nn.Module):
    def __init__(self,dim,max_h=64,max_w=64,rank=8):
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.rank = rank
        max_position = max_h * max_w
        self.position_basis = nn.Parameter(torch.zeros(max_position, rank))
        self.position_projection = nn.Parameter(torch.zeros(rank, dim))
        nn.init.trunc_normal_(self.position_basis,std=0.02)
        nn.init.trunc_normal_(self.position_projection,std=0.02)

    def forward(self, tokens, H, W):
        _, N, D = tokens.shape
        assert H <= self.max_h
        assert W <= self.max_w
        assert N == H * W
        assert D == self.dim
        basis = self.position_basis.reshape(self.max_h,self.max_w,self.rank)
        basis = basis[:H, :W, :]
        basis = basis.reshape(H * W,self.rank)
        pos = basis @ self.position_projection
        pos = pos.unsqueeze(0)
        return tokens + pos


class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim=None,output_dim=None,dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim=input_dim

        if output_dim is None:
            output_dim=input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,output_dim)
        )

    def forward(self,x):
        return self.net(x)

class CrossAttentionTransformer(nn.Module):
    def __init__(self,dim,num_heads=8,dropout=0.0,ffn_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=dim,num_heads=num_heads,dropout=dropout,batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim=dim,hidden_dim=int(dim * ffn_ratio),dropout=dropout)
        
    def forward(self,query_state,visual_tokens):
        attn_output, _ = self.cross_attention(query=query_state,key=visual_tokens,value=visual_tokens,need_weights=False)
        query_state = self.norm1(query_state + attn_output)
        self_attn_output, _ = self.self_attention(query=query_state,key=query_state,value=query_state,need_weights=False)
        query_state = self.norm2(query_state + self_attn_output)
        query_state = self.norm3(query_state + self.ffn(query_state))
        return query_state

class FeedForwardNetwork(nn.Module):
    def __init__(self,dim,hidden_dim=None,dropout=0.0):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim*2

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
