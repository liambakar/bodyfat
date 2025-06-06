import os
from models.vit_model import VideoViT
from utils.dataset import VideoDataset
from utils.training_utils import play_video, get_dataset
import torch
import tqdm

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Running on {device.type}')

    df = get_dataset()

    video_root = os.path.dirname(os.path.dirname(__file__))
    video_root = os.path.join(
        video_root,
        'dataset',
        'videos',
    )
    # video_path = df.iloc[0].loc['video_path']
    # play_video(os.path.join(video_root, video_path))

    BATCH_SIZE = len(df)
    EPOCHS = 2
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 8  # 10%, 20%, ..., 80%
    FRAME_SIZE = 224       # Each frame will be resized to 224x224 pixels
    PATCH_SIZE = 16        # Each patch will be 16x16 pixels
    NUM_FRAMES = 8         # The model expects 8 frames from the video
    IN_CHANNELS = 3        # RGB video
    EMBED_DIM = 768        # Embedding dimension (standard ViT base)
    NUM_HEADS = 12         # Number of attention heads
    NUM_LAYERS = 12        # Number of transformer blocks
    # Example: 10 different video classes (e.g., actions)

    train_dataset = VideoDataset(df)
    dataloader = train_dataset.get_dataloader(batch_size=BATCH_SIZE)

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

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_loss = []
    raise ValueError
    for epoch in tqdm.tqdm(range(EPOCHS), unit='epoch'):

        model.train()
        running_loss = 0.0
        for video_frames, label in dataloader:
            video_frames = video_frames.to(device)
            # video: (batch_size, num_frames, in_channels, H_orig, W_orig)
            label = label.to(device)
            out = model(video_frames)
            print(out)

            loss = loss_func(out, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            train_loss.append(loss.item())

        print(f'Epoch {epoch + 1}/{EPOCHS} Loss: {running_loss}')
