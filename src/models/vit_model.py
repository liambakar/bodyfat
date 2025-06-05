from .modules import PatchEmbedding, TransformerBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from einops import repeat
except ImportError:
    print("Please install einops: pip install einops")
    exit()
try:
    import timm
except ImportError:
    print("Please install timm: pip install timm")
    exit()


class VideoViT(nn.Module):
    """
    Baseline Video Vision Transformer (ViT) model for video classification.
    Handles non-fixed size input videos by resizing frames internally.
    """

    def init_weights(self, pretrained_vit_name, num_spatial_patches_h, num_spatial_patches_w):
        print(
            f"Loading pre-trained 2D ViT weights from '{pretrained_vit_name}'...")
        # Load a pre-trained 2D ViT model from timm
        # Set num_classes=0 to avoid loading the final classification head
        # as we will replace it or fine-tune our own.
        timm_model = timm.create_model(
            pretrained_vit_name, pretrained=True, num_classes=0)

        # --- Initialize Patch Embedding weights ---
        # Pre-trained 2D ViT patch embedding is typically a Conv2d: (out_c, in_c, k_h, k_w)
        # Our 3D patch embedding is a Conv3d: (out_c, in_c, k_t, k_h, k_w)
        # Since our k_t=1, we can directly copy the 2D weights into the spatial dimensions.
        if timm_model.patch_embed.proj.weight.shape[2:] == (self.patch_size, self.patch_size):
            # Copy 2D conv weights to 3D conv weights by unsqueezing temporal dimension
            self.patch_embedding.projection.weight.data = timm_model.patch_embed.proj.weight.unsqueeze(
                2)
            self.patch_embedding.projection.bias.data = timm_model.patch_embed.proj.bias.data
            print("Patch embedding weights loaded.")
        else:
            print(f"Warning: Pre-trained patch size {timm_model.patch_embed.proj.weight.shape[2:]} "
                  f"does not match model patch size ({self.patch_size}, {self.patch_size}). "
                  "Patch embedding will not be loaded from pre-trained model.")

        # --- Initialize CLS token ---
        self.cls_token.data = timm_model.cls_token.data
        print("CLS token loaded.")

        # --- Initialize Positional Embeddings ---
        # Pre-trained pos_embed: (1, num_2d_patches + 1, embed_dim)
        # Our pos_embed: (1, num_frames * num_spatial_patches + 1, embed_dim)
        # We need to adapt the 2D positional embeddings to video.
        # We assume the pre-trained model has 1 CLS token and 2D spatial patches.
        pretrained_2d_pos_embed = timm_model.pos_embed.data
        # CLS token's positional embedding
        pretrained_cls_pos = pretrained_2d_pos_embed[:, 0, :]
        # Spatial positional embeddings
        pretrained_spatial_pos = pretrained_2d_pos_embed[:, 1:, :]

        # Reshape 2D spatial positions to grid for interpolation
        # Assuming square patches for simplicity (H_patches = W_patches)
        sqrt_num_spatial_patches_2d = int(
            math.sqrt(pretrained_spatial_pos.shape[1]))
        pretrained_spatial_pos = pretrained_spatial_pos.reshape(
            1, sqrt_num_spatial_patches_2d, sqrt_num_spatial_patches_2d, self.embed_dim
        ).permute(0, 3, 1, 2)  # (1, embed_dim, H_patches, W_patches)

        # Interpolate to target spatial patch grid size if needed
        if num_spatial_patches_h != sqrt_num_spatial_patches_2d or num_spatial_patches_w != sqrt_num_spatial_patches_2d:
            print(f"Interpolating pre-trained positional embeddings from {sqrt_num_spatial_patches_2d}x{sqrt_num_spatial_patches_2d} "
                  f"to {num_spatial_patches_h}x{num_spatial_patches_w} spatial patches.")
            pretrained_spatial_pos = F.interpolate(
                pretrained_spatial_pos,
                size=(num_spatial_patches_h, num_spatial_patches_w),
                mode='bicubic',
                align_corners=False
            )
        pretrained_spatial_pos = pretrained_spatial_pos.permute(
            0, 2, 3, 1).flatten(1, 2)  # (1, num_spatial_patches, embed_dim)

        # Combine spatial positions for all frames
        # Repeat the spatial positional embeddings num_frames times
        adapted_spatial_pos = repeat(
            pretrained_spatial_pos, '1 n d -> 1 (t n) d', t=self.num_frames)

        # Concatenate the CLS token's positional embedding and the adapted spatial embeddings
        self.positional_embedding.data = torch.cat(
            (pretrained_cls_pos.unsqueeze(1), adapted_spatial_pos), dim=1)
        print("Positional embeddings loaded and adapted.")

        # --- Initialize Transformer Encoder Blocks ---
        # Copy weights from the pre-trained timm model's blocks
        for i, block in enumerate(self.transformer_blocks):
            if i < len(timm_model.blocks):  # Only copy if we have a corresponding block
                print(f"Loading weights for Transformer Block {i}.")
                # Copy attention layer weights
                block.attn.qkv.weight.data = timm_model.blocks[i].attn.qkv.weight.data
                block.attn.qkv.bias.data = timm_model.blocks[i].attn.qkv.bias.data
                block.attn.proj.weight.data = timm_model.blocks[i].attn.proj.weight.data
                block.attn.proj.bias.data = timm_model.blocks[i].attn.proj.bias.data

                # Copy FeedForward (MLP) layer weights
                # fc1 in timm ViT
                block.ffn.net[0].weight.data = timm_model.blocks[i].mlp.fc1.weight.data
                block.ffn.net[0].bias.data = timm_model.blocks[i].mlp.fc1.bias.data
                # fc2 in timm ViT
                block.ffn.net[3].weight.data = timm_model.blocks[i].mlp.fc2.weight.data
                block.ffn.net[3].bias.data = timm_model.blocks[i].mlp.fc2.bias.data

                # Copy LayerNorm weights
                block.norm1.weight.data = timm_model.blocks[i].norm1.weight.data
                block.norm1.bias.data = timm_model.blocks[i].norm1.bias.data
                block.norm2.weight.data = timm_model.blocks[i].norm2.weight.data
                block.norm2.bias.data = timm_model.blocks[i].norm2.bias.data
            else:
                print(
                    f"Warning: No pre-trained weights for Transformer Block {i}. Initializing from scratch.")

        # --- Initialize final normalization layer ---
        if hasattr(timm_model, 'norm') and timm_model.norm is not None:
            self.norm.weight.data = timm_model.norm.weight.data
            self.norm.bias.data = timm_model.norm.bias.data
            print("Final normalization layer weights loaded.")
        else:
            print(
                "Warning: No final normalization layer in pre-trained model to load.")

        # --- Classification head (usually re-initialized for new tasks) ---
        # The final head is typically specific to the pre-training task (ImageNet)
        # and usually needs to be re-initialized for a new task (video classification).
        # We will keep our randomly initialized head for 'num_classes'.
        print(
            "Classification head is randomly initialized for the new task and 'num_classes'.")
        del timm_model  # Release memory

    def __init__(
            self,
            frame_size,         # (H, W) tuple or int, e.g., 224
            patch_size,         # (P, P) tuple or int, e.g., 16
            num_frames,         # Number of frames to process from video, e.g., 8
            in_channels=3,      # Number of input channels (RGB=3)
            embed_dim=768,      # Dimension of the patch embeddings
            num_heads=12,       # Number of attention heads
            num_layers=12,      # Number of Transformer encoder blocks
            mlp_ratio=4.,       # Ratio for FeedForward hidden dimension
            num_classes=1000,   # Number of output classes
            dropout_rate=0.1,   # Dropout rate
            # Pooling type for final output ('cls' token or 'mean' pooling)
            pool='cls',
            pretrained_vit_name=None,
            pretrained=False,   # Whether to load pre-trained weights
    ):
        super().__init__()
        # Ensure frame_size and patch_size are integers if given as tuples
        if isinstance(frame_size, tuple):
            frame_size = frame_size[0]
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        if not (0 <= dropout_rate <= 1):
            raise ValueError("Dropout rate must be between 0 and 1.")

        assert pool in {
            'cls', 'mean'}, "pool type must be either 'cls' (for classification token) or 'mean' (for average pooling of patch embeddings)"

        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool = pool

        # 1. Patch Embedding Layer
        self.patch_embedding = PatchEmbedding(
            frame_size=frame_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames
        )

        num_spatial_patches_h = frame_size // patch_size
        num_spatial_patches_w = frame_size // patch_size
        num_spatial_patches = num_spatial_patches_h * num_spatial_patches_w

        # Total sequence length: (number of frames * spatial patches per frame) + 1 (for CLS token)
        self.sequence_length = num_frames * num_spatial_patches + 1

        # 2. Learnable Classification Token (CLS token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. Learnable Positional Embeddings
        # These embeddings capture the spatial and temporal position of each patch.
        # +1 for the CLS token
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.sequence_length, embed_dim))

        # 4. Dropout for embeddings
        self.dropout = nn.Dropout(dropout_rate)

        # 5. Transformer Encoder Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(num_layers)
        ])

        # 6. Normalization before the classification head
        self.norm = nn.LayerNorm(embed_dim)

        # 7. Output Head
        self.head = nn.Linear(embed_dim, num_classes)

        # 8. Initialize Weights
        self.init_weights(pretrained_vit_name=pretrained_vit_name,
                          num_spatial_patches_h=num_spatial_patches_h,
                          num_spatial_patches_w=num_spatial_patches_w)

    def forward(self, video):
        # video: (batch_size, num_frames, in_channels, H_orig, W_orig)
        # H_orig, W_orig can be non-fixed. The PatchEmbedding layer handles resizing.

        B = video.shape[0]

        # 1. Get patch embeddings and flatten them
        # (batch_size, num_frames * num_spatial_patches, embed_dim)
        x = self.patch_embedding(video)

        # 2. Prepend the learnable CLS token to the sequence
        # Expand CLS token to match batch size
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        # (B, sequence_length, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. Add positional embeddings
        # This handles both spatial and temporal positioning implicitly
        x += self.positional_embedding  # Positional embedding broadcasted across batch

        # 4. Apply dropout
        x = self.dropout(x)

        # 5. Pass through Transformer Encoder Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # 6. Apply final normalization
        x = self.norm(x)

        # 7. Pool the output for classification
        if self.pool == 'cls':
            # Use the output corresponding to the CLS token
            # (batch_size, embed_dim)
            x = x[:, 0]
        elif self.pool == 'mean':
            # Average pool all patch embeddings (excluding CLS token)
            # (batch_size, embed_dim)
            x = x[:, 1:].mean(dim=1)

        # 8. Task head
        output = self.head(x)
        return output
