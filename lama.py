import math
import torch
import torch.nn as nn
import torch.fft as fft
from einops import rearrange
from torchinfo import summary
import random

def get_masked_image(x, h1 = 16, w1 = 16, mask_ratio = 0.75):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, c, h, w = x.shape
    sample_nums = random.sample(range(0, h1 * w1), int(h1 * w1 * (1-mask_ratio)))
    mask = torch.zeros(b, c, h, w).to(device)
    mask_size = h // h1
    for sample in sample_nums:
        hh = sample // w1
        ww = sample % w1
        h_s = hh * mask_size
        h_e = h_s + mask_size
        w_s = ww * mask_size
        w_e = w_s + mask_size
        mask[:, :, h_s:h_e, w_s:w_e] = 1
    return x * mask, mask

class NonLocalAttention(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat) -> torch.Tensor:
        b, c, h, w = feat.shape
        out_feat = self.conv(feat)
        out_feat = rearrange(out_feat, 'b c h w -> b (h w) c')
        out_feat = torch.unsqueeze(out_feat, -1)
        out_feat = self.softmax(out_feat)
        out_feat = torch.squeeze(out_feat, -1)
        identity = rearrange(feat, 'b c h w -> b c (h w)')
        out_feat = torch.matmul(identity, out_feat)
        out_feat = torch.unsqueeze(out_feat, -1)
        return out_feat


class NonLocalAttentionBlock(nn.Module):
    """simplified Non-Local attention network"""

    def __init__(self, in_channels) -> None:
        super().__init__()
        self.nonlocal_attention = NonLocalAttention(in_channels)
        self.global_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, feat):
        out_feat = self.nonlocal_attention(feat)
        out_feat = self.global_transform(out_feat)
        return feat + out_feat


class SpectralTransformer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        out_feat = fft.rfft2(feat, norm='ortho')
        out_feat = torch.cat([out_feat.real, out_feat.imag], dim=1)
        out_feat = self.conv(out_feat)
        out_feat = self.lrelu(out_feat)
        c = out_feat.shape[1]
        out_feat = torch.complex(out_feat[:, : c // 2], out_feat[:, c // 2 :])
        out_feat = fft.irfft2(out_feat, norm='ortho')
        return out_feat


class FourierConvolutionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.half_channels = in_channels // 2
        self.func_g_to_g = SpectralTransformer(self.half_channels)
        self.func_g_to_l = nn.Sequential(
            nn.Conv2d(self.half_channels, self.half_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.func_l_to_g = nn.Sequential(
            nn.Conv2d(self.half_channels, self.half_channels, kernel_size=1),
            NonLocalAttentionBlock(self.half_channels),
        )
        self.func_l_to_l = nn.Sequential(
            nn.Conv2d(self.half_channels, self.half_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        global_feat = feat[:, self.half_channels :]
        local_feat = feat[:, : self.half_channels]
        out_global_feat = self.func_l_to_g(local_feat) + self.func_g_to_g(global_feat)
        out_local_feat = self.func_g_to_l(global_feat) + self.func_l_to_l(local_feat)
        return torch.cat([out_global_feat, out_local_feat], 1)

class ResidualFFC(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.calc = nn.Sequential(
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            FourierConvolutionBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
    def forward(self, x):
        identity = x
        return identity + self.calc(x)

class ResidualBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.compute = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.compute(x)
        identity = self.skip(x)
        out += identity
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.compute = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.compute(x)
        identity = self.skip(x)
        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.compute = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1)

    def forward(self, x):
        identity = x
        out = self.compute(x)
        identity = self.upsample(x)
        out += identity
        return out

class LaMa(nn.Module):
    def __init__(self, mask_ratio = 0.75) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        hidden_dim = 64
        self.encoder = nn.Sequential(
            ResidualBlockWithStride(3, hidden_dim),
            ResidualBlockWithStride(hidden_dim, hidden_dim),
            ResidualBlockWithStride(hidden_dim, hidden_dim),
        )
        self.calc = nn.Sequential(
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
            ResidualFFC(hidden_dim),
        )
        self.decoder = nn.Sequential(
            ResidualBlockUpsample(hidden_dim, hidden_dim),
            ResidualBlockUpsample(hidden_dim, hidden_dim),
            ResidualBlockUpsample(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 3, 3, 1, 1)
        )

    def forward(self, x):
        x, mask = get_masked_image(x, mask_ratio = self.mask_ratio)
        x = self.encoder(x)
        x = self.calc(x)
        x = self.decoder(x)
        return x, mask


if __name__ == "__main__":
    lama = LaMa()
    feat = torch.randn(8, 3, 128, 128)
    summary(lama, input_data=feat)
