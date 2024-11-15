import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class RNN(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            dropout_p=0.1,
            bidirectional=False,
    ):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.dense = nn.Linear(hidden_dim * 2, emb_dim)
        else:
            self.dense = nn.Linear(hidden_dim, emb_dim)
    
    def forward(self, x):
        # x:(b,t,d)
        x,_ = self.rnn(x)
        x = self.dense(x)
        return x


class DualPathRNN(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.intra_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_rnn_attn = RNN(emb_dim, hidden_dim // 2, dropout_p, bidirectional=True)

        self.inter_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_rnn_attn = RNN(emb_dim, hidden_dim, dropout_p, bidirectional=False)


    def forward(self, x):
        # x:(b,d,t,f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)  # (b,t,f,d)

        x_res = x
        x = self.intra_norm(x)
        x = x.reshape(B * T, F, D)  # (b*t,f,d)
        x = self.intra_rnn_attn(x)
        x = x.reshape(B, T, F, D)
        x = x + x_res

        x_res = x
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3)  # (b,f,t,d)
        x = x.reshape(B * F, T, D)
        x = self.inter_rnn_attn(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3) # (b,t,f,d)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x:(b,d,t,f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x


class DPR(nn.Module):
    def __init__(
            self,
            emb_dim=16,
            hidden_dim=24,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=2, dropout_p=dropout_p)

    def forward(self, x):
        x = self.dp_rnn_attn(x)
        x = self.conv_glu(x)
        return x