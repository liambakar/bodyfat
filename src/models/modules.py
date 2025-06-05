import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange
    from einops.layers.torch import EinMix as Rearrange
except ImportError:
    print("Please install einops: pip install einops")
    exit()


class PatchEmbedding(nn.Module):
    """
    Converts video frames into a sequence of flattened patches and projects them
    into a higher-dimensional space (embedding dimension).
    Also handles resizing of input frames.
    """

    def __init__(self, frame_size, patch_size, in_channels, embed_dim, num_frames):
        super().__init__()
        # Validate that frame_size is divisible by patch_size
        if frame_size % patch_size != 0:
            raise ValueError("Frame size must be divisible by patch size.")

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Calculate number of patches along height and width
        self.num_patches_h = frame_size // patch_size
        self.num_patches_w = frame_size // patch_size
        self.num_spatial_patches = self.num_patches_h * self.num_patches_w

        # Calculate the dimension of a flattened patch
        patch_dim = in_channels * patch_size * patch_size

        # Convolutional layer to extract patches and project them to embed_dim
        # This is equivalent to unfolding patches and then linearly projecting
        # using a kernel size equal to patch_size and stride equal to patch_size.
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            # 1 for temporal dimension, patch_size for spatial
            kernel_size=(1, patch_size, patch_size),
            # 1 for temporal dimension, patch_size for spatial
            stride=(1, patch_size, patch_size)
        )

    def forward(self, x):
        # x: (batch_size, num_frames, in_channels, height, width)

        # 1. Resize frames if necessary to a consistent size
        # This is how we handle "non-fixed size" inputs.
        # We assume input frames are B, T, C, H_orig, W_orig
        # We need to resize to B, T, C, self.frame_size, self.frame_size
        B, T, C, H_orig, W_orig = x.shape
        if H_orig != self.frame_size or W_orig != self.frame_size:
            # Reshape for F.interpolate: (B*T, C, H_orig, W_orig)
            x_reshaped = x.view(B * T, C, H_orig, W_orig)
            # Resize
            x_resized = F.interpolate(x_reshaped, size=(
                self.frame_size, self.frame_size), mode='bicubic', align_corners=False)
            # Reshape back to (B, T, C, frame_size, frame_size)
            x = x_resized.view(B, T, C, self.frame_size, self.frame_size)

        # Permute to (batch_size, in_channels, num_frames, height, width) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # Apply 3D convolution to extract patches and project to embed_dim
        # Output will be (B, embed_dim, T_patches, H_patches, W_patches)
        # where T_patches = num_frames / 1 = num_frames (since temporal stride is 1)
        x = self.projection(x)

        # 'b d t h w -> b (t h w) d':
        # b: batch_size
        # d: embed_dim (depth/channels of the projection output)
        # t: num_frames (temporal patches, which is simply num_frames here)
        # h: num_patches_h (spatial patches height)
        # w: num_patches_w (spatial patches width)
        # (t h w): flattened sequence of all patches
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        return x


class Attention(nn.Module):
    """
    Multi-head Self-Attention (MSA) module.
    """

    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores

        # Linear layers for Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch_size, sequence_length, embed_dim)
        B, N, C = x.shape

        # Project Q, K, V from input tensor and separate heads
        # qkv_output: (B, N, embed_dim * 3)
        # rearrange: 'b n (qkv h d) -> qkv b h n d'
        # qkv: 3 (for Q, K, V)
        # h: num_heads
        # d: head_dim
        # Split into (Q, K, V) each (B, N, embed_dim)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # Calculate attention scores: Q @ K_T * scale
        # matmul(q, k.transpose(-1, -2)): (B, num_heads, N, N)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)  # Apply softmax for probabilities
        attn = self.attn_drop(attn)

        # Multiply attention weights with Value: (B, num_heads, N, head_dim)
        x = attn @ v
        # Concatenate heads and project back to embed_dim
        # rearrange: 'b h n d -> b n (h d)'
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP) within the Transformer block.
    """

    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # Gaussian Error Linear Unit activation
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder Block, combining Multi-head Self-Attention
    and a Feed-Forward network, with Layer Normalization and Residual Connections.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Hidden dimension for MLP is mlp_ratio * embed_dim
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout_rate)

    def forward(self, x):
        # x + MSA(LayerNorm(x))
        x = x + self.attn(self.norm1(x))
        # x + FFN(LayerNorm(x))
        x = x + self.ffn(self.norm2(x))
        return x
