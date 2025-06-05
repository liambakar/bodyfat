import torch
from models.vit_model import VideoViT

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Running on {device.type}')

    # Define model parameters
    FRAME_SIZE = 224       # Each frame will be resized to 224x224 pixels
    PATCH_SIZE = 16        # Each patch will be 16x16 pixels
    NUM_FRAMES = 8         # The model expects 8 frames from the video
    IN_CHANNELS = 3        # RGB video
    EMBED_DIM = 768        # Embedding dimension (standard ViT base)
    NUM_HEADS = 12         # Number of attention heads
    NUM_LAYERS = 12        # Number of transformer blocks
    # Example: 10 different video classes (e.g., actions)
    NUM_CLASSES = 10

    # Instantiate the VideoViT model
    model = VideoViT(
        frame_size=FRAME_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        pretrained_vit_name='vit_base_patch16_224',
        pretrained=True,
    ).to(device)
    print("Model initialized successfully!")
    print(
        f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Scenario 1: Input video with frames exactly matching FRAME_SIZE
    batch_size_1 = 2
    # (batch_size, num_frames, in_channels, height, width)
    dummy_video_1 = torch.randn(
        batch_size_1, NUM_FRAMES, IN_CHANNELS, FRAME_SIZE, FRAME_SIZE).to(device)
    print(f"\nInput video shape (fixed size): {dummy_video_1.shape}")
    output_1 = model(dummy_video_1)
    # Should be (batch_size, num_classes)
    print(f"Output logits shape (fixed size): {output_1}")

    # Scenario 2: Input video with frames of a *different* size (e.g., smaller)
    batch_size_2 = 1
    H_small, W_small = 112, 112  # Smaller frame size
    dummy_video_2 = torch.randn(
        batch_size_2, NUM_FRAMES, IN_CHANNELS, H_small, W_small).to(device)
    print(f"\nInput video shape (smaller size): {dummy_video_2.shape}")
    output_2 = model(dummy_video_2)
    print(f"Output logits shape (smaller size): {output_2}")

    # Scenario 3: Input video with frames of another *different* size (e.g., larger)
    batch_size_3 = 3
    H_large, W_large = 300, 400  # Larger and rectangular frame size
    dummy_video_3 = torch.randn(
        batch_size_3, NUM_FRAMES, IN_CHANNELS, H_large, W_large).to(device)
    print(
        f"\nInput video shape (larger, rectangular size): {dummy_video_3.shape}")
    output_3 = model(dummy_video_3)
    print(f"Output logits shape (larger, rectangular size): {output_3.shape}")

    print("\nBaseline Video ViT model successfully demonstrated with non-fixed input sizes.")
