import torch
from torch import nn
from torch import Tensor
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncoding (nn.Module):
    '''
    - fixed positional embedding (there's also other type of embedding where the parameters are learnable)
    - emb_{2i} = sin(pos / 10000^(2i / d_model))
    - emb_{2i+1} = cos(pos / 1000^(2i / d_model))
    '''
    def __init__(self, d_model, dropout=0.1, max_len=2048) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.pos_emb = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(dim=1)   # (2048, 1)
        _2i = torch.arange(start=0, end=d_model, step=2)
        
        self.pos_emb[:, 0::2] = np.sin(position / 10000**(_2i / d_model))
        self.pos_emb[:, 1::2] = np.cos(position / 10000**(_2i / d_model))
        self.pos_emb = self.pos_emb.unsqueeze(dim=0).to(device)

    def forward(self, X: Tensor):
        '''X.shape = [seq_len, batch_size, d_model]'''
        with torch.no_grad():
            # print(X.shape, self.pos_emb[:, :X.shape[1], :].shape)
            # X = X + self.pos_emb[:, :X.shape[1], :]     # d_model has already been specified
            # X = X*np.sqrt(self.d_model) + self.pos_emb[:, :X.shape[1]]
            b, len, d_model = X.shape
            ones = torch.ones((b, len, d_model)).to(device)
            # out = self.dropout(X)
            out = self.pos_emb[:, :X.shape[1], :] * ones
        return out

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)
    
