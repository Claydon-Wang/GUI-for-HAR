import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from torch.nn.modules.utils import _pair

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLP_WISDM(nn.Module):

    def __init__(self, in_channels=1, dim=256, num_classes=6, patch_size=(10, 3), image_size=(200, 3), depth=1, token_dim=256, channel_dim=768, dropout=0.01):
        super().__init__()
        self.image_size = _pair(image_size)
        self.patch_size = _pair(patch_size)

        image_size_h = image_size[0]
        image_size_w = image_size[1]
        patch_size_h = patch_size[0]
        patch_size_w = patch_size[1]

        assert image_size_h % patch_size_h == 0, 'Image_h dimensions must be divisible by the patch_size_h.'
        assert image_size_w % patch_size_w == 0, 'Image_h dimensions must be divisible by the patch size_h.'
        num_patch_h = (image_size_h // patch_size_h)
        num_patch_w = (image_size_w // patch_size_w)
        self.num_patch = num_patch_h * num_patch_w
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_size_h, p2=patch_size_w),
            nn.Linear(in_channels * patch_size_h * patch_size_w, dim)
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim, dropout=dropout))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):


        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)














