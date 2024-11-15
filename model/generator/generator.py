from .dpr_layer import DPR, CustomLayerNorm
import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks



class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3)),
        )
        self.high_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 5), stride=(1, 3)),
        )
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)
    
    def forward(self, x):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]
        
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)

        x = torch.cat([x_low, x_high], dim=-1)
        x = self.norm(x)
        x = self.act(x)
        return x


class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
        )
        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(1, 3), r=3)
    
    def forward(self, x):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]

        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class NoiseDetector(nn.Module):
    def __init__(
            self,
            in_channels=1,
            emb_dim=16,
            hidden_dim=16 * 2,
            n_freqs=64,
            dropout_p=0.1,
    ):
        super().__init__()
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, emb_dim//4, (1, 1), (1, 1)),
            CustomLayerNorm((1, n_freqs), stat_dims=(1, 3)),
            nn.PReLU(emb_dim//4),
        )
        self.conv_2 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim//4, emb_dim//2, (2, 3), (1, 2)), # 32
            CustomLayerNorm((1, n_freqs//2), stat_dims=(1, 3)),
            nn.PReLU(emb_dim//2),
        )
        self.conv_3 = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim//2, emb_dim, (2, 3), (1, 2)),  # 16
            CustomLayerNorm((1, n_freqs//4), stat_dims=(1, 3)),
            nn.PReLU(emb_dim),
        )

        self.dpr = DPR(emb_dim, hidden_dim, n_freqs=n_freqs//4)

        self.down = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(emb_dim, emb_dim * 2, (2, 3), (1, 2)),  # 8
            CustomLayerNorm((1, n_freqs//8), stat_dims=(1, 3)),
            nn.PReLU(emb_dim * 2),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear_block = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.PReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(emb_dim * 2, 1),
        )

    
    def forward(self, x):
        # x:(b,d,t,f)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        
        x = self.dpr(x)

        x = self.down(x)  # (b,d,t,f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B*T, D, F)
        x = self.pool(x)  # (b*t,d,1)
        x = x.squeeze(-1).reshape(B, T, D)

        x = self.linear_block(x).squeeze(-1)  # (b,t)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels//4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        
        self.conv_2 = DSConv(num_channels//4, num_channels//2, n_freqs=257)
        self.conv_3 = DSConv(num_channels//2, num_channels//4*3, n_freqs=128)
        self.conv_4 = DSConv(num_channels//4*3, num_channels, n_freqs=64)


    def forward(self, x):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x)
        out_list.append(x)  # 128
        x = self.conv_3(x)
        out_list.append(x)  # 64
        x = self.conv_4(x)
        out_list.append(x)  # 32
        return out_list


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1):
        super(MaskDecoder, self).__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64)  # 128
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128)  # 256
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(num_channels // 4, out_channel, (2, 2)), # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x, encoder_out_list):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1))  # 64
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1))  # 128
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1))  # 256
        x = self.mask_conv(x)  # (B,out_channel,T,F)
        x = x.permute(0, 3, 2, 1)  # (B,F,T,out_channel)
        x = self.lsigmoid(x).permute(0, 3, 2, 1)
        return x


class LiSenNet(nn.Module):
    def __init__(self, num_channels=16, n_blocks=2, n_fft=512, hop_length=256, compress_factor=0.3):
        super(LiSenNet, self).__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.hop_length = hop_length
        self.compress_factor = compress_factor

        self.encoder = Encoder(in_channels=3, num_channels=num_channels)

        self.blocks = nn.Sequential(
            *[DPR(
                emb_dim=num_channels,
                hidden_dim=num_channels // 2 * 3,
                n_freqs=self.n_freqs // (2 ** 3),
                dropout_p=0.1,
            ) for _ in range(n_blocks)]
        )

        self.decoder = MaskDecoder(self.n_freqs, num_channels=num_channels, out_channel=2, beta=1)

    def apply_stft(self, x, return_complex=True):
        # x:(B,T)
        assert x.ndim == 2
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            return_complex=return_complex,
        ).transpose(1, 2)  # (B,T,F)
        return spec

    def apply_istft(self, x, length=None):
        # x:(B,T,F)
        assert x.ndim == 3
        x = x.transpose(1, 2)  # (B,F,T)
        audio = torch.istft(
            x,
            self.n_fft,
            self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            onesided=True,
            length=length,
            return_complex=False
        )  # (B,T)
        return audio

    def power_compress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** self.compress_factor
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def power_uncompress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** (1.0 / self.compress_factor)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    
    def mel_scale(self, mag, sr=16000, f_min=0.0, f_max=8000.0, n_mels=64):
        if not hasattr(self, 'fb'):
            fb = melscale_fbanks(n_freqs=self.n_freqs, f_min=f_min, f_max=f_max, n_mels=n_mels, sample_rate=sr)
            setattr(self, 'fb', fb.to(mag.device))
        mag = mag ** (1 / self.compress_factor)  # uncompress
        mel = mag @ self.fb  # (B,T,M)
        mel = mel ** self.compress_factor  # compress
        return mel

    @staticmethod
    def cal_gd(x):
        # x: (B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_gd = torch.diff(x, dim=2, prepend=torch.zeros(b, t, 1, device=x.device))  # (-2pi, 2pi]
        return torch.atan2(x_gd.sin(), x_gd.cos())

    @staticmethod
    def cal_if(x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  # (-2pi, 2pi]
        return torch.atan2(x_if.sin(), x_if.cos())
    
    def cal_ifd(self, x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  # (-2pi, 2pi]
        x_ifd = x_if - 2 * torch.pi * (self.hop_length / self.n_fft) * torch.arange(f, device=x.device)[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())

    def griffinlim(self, mag, pha=None, length=None, n_iter=2, momentum=0.99):
        mag = mag.detach()
        mag = mag ** (1.0 / self.compress_factor) # uncompress
        assert 0 <= momentum < 1
        momentum = momentum / (1 + momentum)
        if pha is None:
            pha = torch.rand(mag.size(), dtype=mag.dtype, device=mag.device)

        tprev = torch.tensor(0.0, dtype=mag.dtype, device=mag.device)
        for _ in range(n_iter):
            inverse = self.apply_istft(torch.complex(mag * pha.cos(), mag * pha.sin()), length=length)
            rebuilt = self.apply_stft(inverse)
            pha = rebuilt
            pha = pha - tprev.mul_(momentum)
            pha = pha.angle()
            tprev = rebuilt

        return pha

    def forward(self, src, tgt=None):
        if tgt == None:
            tgt = src
        src_spec = self.power_compress(self.apply_stft(src))  # (B,T,F)
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()
        src_gd = self.cal_gd(src_pha)
        src_ifd = self.cal_ifd(src_pha)

        tgt_spec = self.power_compress(self.apply_stft(tgt))  # (B,T,F)
        tgt_mag = tgt_spec.abs()

        x = torch.stack([src_mag, src_gd / torch.pi, src_ifd / torch.pi], dim=1)  # (B,3,T,F)

        encoder_out_list = self.encoder(x)
        x = self.blocks(encoder_out_list[-1])
        x = self.decoder(x, encoder_out_list)  # (B,2,T,F)

        est_mag = (x[:, 0] + 1e-8) * src_mag + (x[:, 1] + 1e-8) * src_mag

        est_pha = self.griffinlim(est_mag.detach(), src_pha, tgt.size(-1))
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est = self.apply_istft(self.power_uncompress(est_spec), length=tgt.size(-1))

        results = {
            'tgt': tgt,
            'tgt_spec': tgt_spec,
            'tgt_mag': tgt_mag,
            'est': est,
            'est_spec': est_spec,
            'est_mag': est_mag,
        }

        return results
