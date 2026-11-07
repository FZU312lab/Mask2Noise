import torch
from torch import nn
from basic_layers import MLP


class ResidualConvHead(nn.Module):
    def __init__(self,in_channels,hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels

        self.head = nn.Sequential(
            nn.Conv2d(in_channels,hidden_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels,1,kernel_size=3,padding=1)
        )

    def forward(self,x):
        return self.head(x)


class ResidualPredictionBlock(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.block = nn.Conv2d(in_channels,1,kernel_size=1)
        
    def forward(self,x):
        return self.block(x)


class QFRR(nn.Module):
    def __init__(self,dim,query_dim=None):
        super().__init__()
        if query_dim is None:
            query_dim = dim
        self.dim = dim
        self.query_score = nn.Sequential(nn.LayerNorm(query_dim),nn.Linear(query_dim,1))
        self.film_generator = MLP(input_dim=query_dim,hidden_dim=query_dim,output_dim=dim * 2)
        self.base_branch = ResidualPredictionBlock(dim)
        self.cond_branch = ResidualPredictionBlock(dim)

    def forward(self,feature,query_state):
        B,Nq,D = query_state.shape
        scores = self.query_score(query_state)
        weights = torch.softmax(scores,dim=1)
        z_g = torch.sum(weights *query_state,dim=1)
        film_params = self.film_generator(z_g)
        gamma,beta = torch.chunk(film_params,2,dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        conditioned_feature = ((1 + gamma) * feature + beta)
        base_residual = self.base_branch(feature)
        cond_residual = self.cond_branch(conditioned_feature)
        residual = (base_residual+cond_residual)
        return residual
