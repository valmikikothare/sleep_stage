import numpy as np
import torch
import torch.nn as nn

from utils import get_activation_class


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.activation = get_activation_class(activation)()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        res = self.downsample(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.activation(x)
        x = self.conv2(x)
        return x + res


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, do_ln, activation):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if do_ln else nn.Identity()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation_class(activation)()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


def positional_encoding(length, depth):
    original_depth = depth
    if depth % 2 != 0:
        depth += 1
    depth /= 2

    positions = np.arange(length)[:, np.newaxis]  # (length, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rads = positions / (10000**depths)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)[
        :, :original_depth
    ]

    return torch.Tensor(pos_encoding)


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_length, out_dim, ff_dim, activation):
        super().__init__()
        pos_encoding = positional_encoding(seq_length, out_dim)[None, :, :]
        self.pos_encoding = nn.Parameter(pos_encoding, requires_grad=False)
        self.ffn = FeedForward(out_dim, ff_dim, out_dim, False, activation)

    def forward(self, x):
        x = x.transpose(1, 2)
        x += self.ffn(self.pos_encoding)
        return x.transpose(1, 2)


class AttentionBlock(nn.Module):
    def __init__(
        self, in_dim, embed_dim, num_heads, feedforward_dim, activation, dropout
    ):
        super().__init__()
        self.qkv = nn.Linear(in_dim, embed_dim * 3)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = FeedForward(
            in_dim, feedforward_dim, in_dim, True, activation
        )
        self.proj = (
            nn.Linear(embed_dim, in_dim) if in_dim != embed_dim else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        res = x
        x = self.norm1(x)
        q, k, v = self.qkv(x).chunk(3, -1)
        x, _ = self.attention(q, k, v, need_weights=False)
        x = self.proj(x)
        x += res
        res = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x += res
        return x.transpose(1, 2)


class AttnNet(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        in_dim,
        out_dim,
        embed_dim,
        feedforward_dim,
        do_pos_embed,
        pos_embed_dim,
        dim_blocks,
        res_blocks,
        attn_blocks,
        num_heads,
        kernel_size,
        activation,
        dropout,
    ):
        super().__init__()
        assert len(dim_blocks) == len(attn_blocks) == len(res_blocks)
        self.num_blocks = len(dim_blocks)
        res_blocks = (
            res_blocks
            if isinstance(res_blocks, list)
            else [res_blocks] * self.num_blocks
        )
        attn_blocks = (
            attn_blocks
            if isinstance(attn_blocks, list)
            else [attn_blocks] * self.num_blocks
        )

        self.pos_embed = (
            PositionalEmbedding(seq_length, in_dim, pos_embed_dim, activation)
            if do_pos_embed
            else nn.Identity()
        )

        self.blocks = nn.ModuleList()
        for i, (dim_block, do_res, do_attn) in enumerate(
            zip(dim_blocks, res_blocks, attn_blocks)
        ):
            assert do_res or do_attn
            block = nn.Module()
            block.res = (
                ResidualBlock(
                    in_dim if i == 0 else dim_blocks[i - 1],
                    dim_block,
                    kernel_size,
                    activation,
                )
                if do_res
                else nn.Identity()
            )
            block.attn = (
                AttentionBlock(
                    dim_block,
                    embed_dim,
                    num_heads,
                    feedforward_dim,
                    activation,
                    dropout,
                )
                if do_attn
                else nn.Identity()
            )
            self.blocks.append(block)

        self.out_block = nn.Sequential(
            nn.Linear(seq_length * dim_blocks[-1], 64),
            get_activation_class(activation)(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block.res(x)
            x = block.attn(x)
        return self.out_block(x.flatten(1))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.bn2(x)
        x = self.conv2(x)
        x += self.shortcut(identity)
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        num_layers,
        hidden_dims,
        blocks,
        strides,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_dims[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                self._make_layer(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    blocks[i],
                    strides[i],
                )
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dims[-1], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(
                BasicBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride if i == 0 else 1,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
